#include "ekf.hpp"
#include <cuda_runtime.h>

// CUDA kernels
__global__ void predictKernel(float *d_state, float *d_covariance, int state_dim, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= state_dim / 2) {
        // Implement the CUDA kernel for the prediction step (dummy update)
        // d_state[i] += d_state[i + state_dim / 2] * dt;
    }
}

__global__ void updateKernel(float *d_state, float *d_covariance, const float *d_measurement, const float *d_H, const float *d_F, float *d_R, int state_dim, int meas_dim, float dt) 
{
    // // // Assuming state_dim and meas_dim are such that the memory for d_covariance is properly allocated
    // extern __shared__ float shared_mem[];

    // int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // // Define constants
    // float identity_value = 1.0f;

    // // Shared memory allocation
    // float *shared_H = shared_mem;
    // float *shared_R = shared_H + state_dim * meas_dim;
    // float *shared_K = shared_R + meas_dim * meas_dim;
    // float *shared_temp = shared_K + state_dim * meas_dim;
    
    // if (tid < meas_dim * state_dim) {
    //     shared_H[tid] = d_H[tid];
    // }
    // if (tid < meas_dim * meas_dim) {
    //     shared_R[tid] = d_R[tid];
    // }

    // __syncthreads();

    // // Compute Kalman Gain (K = P * H^T * (H * P * H^T + R)^-1)
    // if (tid < state_dim * meas_dim) {
    //     shared_K[tid] = 0.0f;  // Initialize K
    //     for (int k = 0; k < meas_dim; ++k) {
    //         shared_K[tid] += d_covariance[tid / state_dim * meas_dim + k] * shared_H[k * state_dim + (tid % state_dim)];
    //     }
    // }

    // __syncthreads();

    // // Compute K * H
    // if (tid < state_dim * meas_dim) {
    //     float sum = 0.0f;
    //     for (int k = 0; k < meas_dim; ++k) {
    //         sum += shared_K[k * state_dim + (tid % state_dim)] * shared_H[k * state_dim + (tid % state_dim)];
    //     }
    //     shared_temp[tid] = sum + shared_R[tid];
    // }

    // __syncthreads();

    // // Compute the inverse of (H * P * H^T + R) using a simple method for demonstration
    // if (tid < meas_dim * meas_dim) {
    //     float inv = 1.0f / shared_temp[tid];
    //     shared_temp[tid] = inv;
    // }

    // __syncthreads();

    // // Compute Kalman Gain (K = K * (H * P * H^T + R)^-1)
    // if (tid < state_dim * meas_dim) {
    //     shared_K[tid] *= shared_temp[tid % meas_dim];
    // }

    // __syncthreads();

    // // Update state (x = x + K * (z - H * x))
    // if (tid < state_dim) {
    //     float y = d_measurement[tid] - d_state[tid];
    //     for (int k = 0; k < meas_dim; ++k) {
    //         y -= shared_K[k * state_dim + tid] * d_measurement[k];
    //     }
    //     d_state[tid] += y;
    // }

    // __syncthreads();

    // // Update covariance (P = (I - K * H) * P)
    // if (tid < state_dim * state_dim) {
    //     float sum = 0.0f;
    //     for (int k = 0; k < meas_dim; ++k) {
    //         sum += shared_K[k * state_dim + (tid % state_dim)] * shared_H[k * state_dim + (tid % state_dim)];
    //     }
    //     d_covariance[tid] = (identity_value - sum) * d_covariance[tid];
    // }
}

// CUDA kernel launches
void EKF::launchPredictKernel() {
    float dt = 1.0f / 30;
    predictKernel<<<1, state_dim_>>>(d_state_, d_covariance_, state_dim_, dt);
}

void EKF::launchUpdateKernel(float *d_measurement, float *d_H, float *d_F, float *d_R, int state_dim, int meas_dim, float dt) {
    int shared_mem_size = (state_dim * meas_dim * 2 + meas_dim * meas_dim) * sizeof(float);
    updateKernel<<<1, state_dim_, shared_mem_size>>>(d_state_, d_covariance_, d_measurement, d_H, d_F, d_R, state_dim, meas_dim, dt);
}
