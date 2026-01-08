#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum[2] = {0, 0};
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 val_square = make_float4(val.x * val.x, val.y * val.y, val.z * val.z, val.w * val.w);
    l_sum[0] += val.x + val.y + val.z + val.w;
    l_sum[1] += val_square.x + val_square.y + val_square.z + val_square.w;
  }

  // Step 2
  warpReduce<ReduceType::kSum, 2>(l_sum);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = l_sum[0] / (hidden_size * 4);
    s_var = l_sum[1] / (hidden_size * 4) - s_mean * s_mean;
    means[blockIdx.x] = s_mean;
    vars[blockIdx.x] = s_var;
  }
  __syncthreads();

  // Step 3
  float rstd = rsqrt(s_var + LN_EPSILON);
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;  
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    ln_res_f4[idx] = make_float4(
                      (val.x - s_mean) * rstd * scale_f4[idx].x + bias_f4[idx].x,
                      (val.y - s_mean) * rstd * scale_f4[idx].y + bias_f4[idx].y,
                      (val.z - s_mean) * rstd * scale_f4[idx].z + bias_f4[idx].z,
                      (val.w - s_mean) * rstd * scale_f4[idx].w + bias_f4[idx].w);
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbeta
Layer norm backword kernel, compute the gradient of gamma and beta.
dbeta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - beta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
beta_grad: [hidden_size], gradient of beta
out_grad: [batch_size * seq_len, hidden_size], gradient of beta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
beta: [hidden_size], beta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && beta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbeta(T *gamma_grad, T *beta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *beta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float beta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1
  T gamma_sum = 0.0f;
  T beta_sum = 0.0f;
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    int col = blockIdx.x * TILE_DIM + threadIdx.y;
    if (col < width) {
      int offset = row * width + col;
      T beta = out_grad[offset];
      T gamma = beta * (inp[offset] - means[row]) * rsqrtf(vars[row]);
      gamma_sum += gamma;
      beta_sum += beta;
    }
  }

  // Step 2
  gamma_buffer[threadIdx.x][threadIdx.y] = gamma_sum;
  beta_buffer[threadIdx.x][threadIdx.y] = beta_sum;
  
  // Step 3
  for (int i = TILE_DIM / 2; i > 0; i >>= 1) {
    gamma_buffer[threadIdx.x][threadIdx.y] += g.shfl_down(gamma_buffer[threadIdx.x][threadIdx.y], i);
    beta_buffer[threadIdx.x][threadIdx.y] += g.shfl_down(beta_buffer[threadIdx.x][threadIdx.y], i);
  }

  __syncthreads();
  
  // Step 4
  if (threadIdx.x == 0) {
    int col = blockIdx.x * TILE_DIM + threadIdx.y;
    if (col < width) {
      gamma_grad[col] = gamma_buffer[0][threadIdx.y];
      beta_grad[col] = beta_buffer[0][threadIdx.y];
    }
  }

  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - beta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of beta ln input
out_grad: [batch_size * seq_len, hidden_size], gradient of beta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
beta: [hidden_size], beta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *beta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  // Step 1, 2
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;  
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;  
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  float4 dxhat{0, 0, 0, 0};
  float4 xhat{0, 0, 0, 0};
  float4 prod{0, 0, 0, 0};
  T mean = means[blockIdx.x];
  T rstd = rsqrtf(vars[blockIdx.x] + LN_EPSILON);
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 inp_val = inp_f4[idx];
    float4 out_grad_val = out_grad_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    dxhat = make_float4(out_grad_val.x * gamma_val.x,
                                out_grad_val.y * gamma_val.y,
                                out_grad_val.z * gamma_val.z,
                                out_grad_val.w * gamma_val.w);
    xhat = make_float4((inp_val.x - mean) * rstd,
                              (inp_val.y - mean) * rstd,
                              (inp_val.z - mean) * rstd,
                              (inp_val.w - mean) * rstd);
    prod = make_float4(dxhat.x * xhat.x,
                      dxhat.y * xhat.y,
                      dxhat.z * xhat.z,
                      dxhat.w * xhat.w);
  }
  // Step 3, 4
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;  
  float reduce_dxhat = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
  float reduce_prod = prod.x + prod.y + prod.z + prod.w;
  float reduce_val[2] = {reduce_dxhat, reduce_prod};
  warpReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float reduce_buf_dxhat[32];
  __shared__ float reduce_buf_prod[32];
  if (threadIdx.x % 32 == 0) {
    reduce_buf_dxhat[threadIdx.x / 32] = reduce_val[0];
    reduce_buf_prod[threadIdx.x / 32] = reduce_val[1];
  }
  __syncthreads();
  float buf[2] = {0.0, 0.0};
  if (threadIdx.x < (blockDim.x + 32 - 1) / 32) {
    buf[0] = reduce_buf_dxhat[threadIdx.x];
    buf[1] = reduce_buf_prod[threadIdx.x];
  }
  warpReduce<ReduceType::kSum, 2>(buf);
  if (threadIdx.x == 0) {
    reduce_buf_dxhat[0] = buf[0];
    reduce_buf_prod[1] = buf[1];
  }
  __syncthreads();

  float dxhat_sum = reduce_buf_dxhat[0];
  float prod_sum = reduce_buf_prod[1];
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dinp_val = make_float4(
                    rstd * (dxhat.x - (dxhat_sum + xhat.x * prod_sum) / (hidden_dim * 4)),
                    rstd * (dxhat.y - (dxhat_sum + xhat.y * prod_sum) / (hidden_dim * 4)),
                    rstd * (dxhat.z - (dxhat_sum + xhat.z * prod_sum) / (hidden_dim * 4)),
                    rstd * (dxhat.w - (dxhat_sum + xhat.w * prod_sum) / (hidden_dim * 4))
                  );
    inp_grad_f4[idx] = dinp_val;
  }
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *beta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *beta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_beta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_beta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_beta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_beta_size);
  cudaMalloc((void **)&d_beta_grad, gamma_beta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_beta_size);
  cudaMalloc((void **)&d_beta, gamma_beta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_beta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_beta, beta, gamma_beta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and beta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbeta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_beta_grad, d_out_grad, d_inp, d_gamma, d_beta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_beta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_beta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(beta_grad, d_beta_grad, gamma_beta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_beta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_beta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
