#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "squint/tensor/cuda/element_wise.hpp"

// CUDA kernel for element-wise addition with different strides
template <typename T>
__global__ void element_wise_addition_kernel(T *output, const T *a, const T *b, const unsigned long *dims,
                                             const unsigned long *strides_out, const unsigned long *strides_a,
                                             const unsigned long *strides_b, unsigned long num_dims,
                                             unsigned long total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int flat_index_out = 0;
        int flat_index_a = 0;
        int flat_index_b = 0;
        int temp = idx;
        for (int i = num_dims - 1; i >= 0; --i) {
            int dim_idx = temp % dims[i];
            flat_index_out += dim_idx * strides_out[i];
            flat_index_a += dim_idx * strides_a[i];
            flat_index_b += dim_idx * strides_b[i];
            temp /= dims[i];
        }
        output[flat_index_out] = a[flat_index_a] + b[flat_index_b];
    }
}

template <typename T>
void element_wise_addition(T *output, const T *a, const T *b, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    element_wise_addition_kernel<<<num_blocks, block_size>>>(output, a, b, dims, strides_out, strides_a, strides_b,
                                                             num_dims, total_size);
}

template void element_wise_addition<float>(float *output, const float *a, const float *b, const unsigned long *dims,
                                           const unsigned long *strides_out, const unsigned long *strides_a,
                                           const unsigned long *strides_b, unsigned long num_dims,
                                           unsigned long total_size);

template void element_wise_addition<double>(double *output, const double *a, const double *b, const unsigned long *dims,
                                            const unsigned long *strides_out, const unsigned long *strides_a,
                                            const unsigned long *strides_b, unsigned long num_dims,
                                            unsigned long total_size);

// CUDA kernel for element-wise subtraction with different strides
template <typename T>
__global__ void element_wise_subtraction_kernel(T *output, const T *a, const T *b, const unsigned long *dims,
                                                const unsigned long *strides_out, const unsigned long *strides_a,
                                                const unsigned long *strides_b, unsigned long num_dims,
                                                unsigned long total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int flat_index_out = 0;
        int flat_index_a = 0;
        int flat_index_b = 0;
        int temp = idx;
        for (int i = num_dims - 1; i >= 0; --i) {
            int dim_idx = temp % dims[i];
            flat_index_out += dim_idx * strides_out[i];
            flat_index_a += dim_idx * strides_a[i];
            flat_index_b += dim_idx * strides_b[i];
            temp /= dims[i];
        }
        output[flat_index_out] = a[flat_index_a] - b[flat_index_b];
    }
}

template <typename T>
void element_wise_subtraction(T *output, const T *a, const T *b, const unsigned long *dims,
                              const unsigned long *strides_out, const unsigned long *strides_a,
                              const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    element_wise_subtraction_kernel<<<num_blocks, block_size>>>(output, a, b, dims, strides_out, strides_a, strides_b,
                                                                num_dims, total_size);
}

template void element_wise_subtraction<float>(float *output, const float *a, const float *b, const unsigned long *dims,
                                              const unsigned long *strides_out, const unsigned long *strides_a,
                                              const unsigned long *strides_b, unsigned long num_dims,
                                              unsigned long total_size);

template void element_wise_subtraction<double>(double *output, const double *a, const double *b, const unsigned long *dims,
                                               const unsigned long *strides_out, const unsigned long *strides_a,
                                               const unsigned long *strides_b, unsigned long num_dims,
                                               unsigned long total_size);

// CUDA kernel for element-wise equality with different strides
template <typename T>
__global__ void element_wise_equality_kernel(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                                             const unsigned long *strides_out, const unsigned long *strides_a,
                                             const unsigned long *strides_b, unsigned long num_dims,
                                             unsigned long total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int flat_index_out = 0;
        int flat_index_a = 0;
        int flat_index_b = 0;
        int temp = idx;
        for (int i = num_dims - 1; i >= 0; --i) {
            int dim_idx = temp % dims[i];
            flat_index_out += dim_idx * strides_out[i];
            flat_index_a += dim_idx * strides_a[i];
            flat_index_b += dim_idx * strides_b[i];
            temp /= dims[i];
        }
        output[flat_index_out] = a[flat_index_a] == b[flat_index_b];
    }
}

template <typename T>
void element_wise_equality(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    element_wise_equality_kernel<<<num_blocks, block_size>>>(output, a, b, dims, strides_out, strides_a, strides_b,
                                                             num_dims, total_size);
}

template void element_wise_equality<float>(uint8_t *output, const float *a, const float *b, const unsigned long *dims,
                                           const unsigned long *strides_out, const unsigned long *strides_a,
                                           const unsigned long *strides_b, unsigned long num_dims,
                                           unsigned long total_size);

template void element_wise_equality<double>(uint8_t *output, const double *a, const double *b, const unsigned long *dims,
                                            const unsigned long *strides_out, const unsigned long *strides_a,
                                            const unsigned long *strides_b, unsigned long num_dims,
                                            unsigned long total_size);

// CUDA kernel for element-wise inequality with different strides
template <typename T>
__global__ void element_wise_inequality_kernel(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                                               const unsigned long *strides_out, const unsigned long *strides_a,
                                               const unsigned long *strides_b, unsigned long num_dims,
                                               unsigned long total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int flat_index_out = 0;
        int flat_index_a = 0;
        int flat_index_b = 0;
        int temp = idx;
        for (int i = num_dims - 1; i >= 0; --i) {
            int dim_idx = temp % dims[i];
            flat_index_out += dim_idx * strides_out[i];
            flat_index_a += dim_idx * strides_a[i];
            flat_index_b += dim_idx * strides_b[i];
            temp /= dims[i];
        }
        output[flat_index_out] = a[flat_index_a] != b[flat_index_b];
    }
}

template <typename T>
void element_wise_inequality(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                             const unsigned long *strides_out, const unsigned long *strides_a,
                             const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    element_wise_inequality_kernel<<<num_blocks, block_size>>>(output, a, b, dims, strides_out, strides_a, strides_b,
                                                               num_dims, total_size);
}

template void element_wise_inequality<float>(uint8_t *output, const float *a, const float *b, const unsigned long *dims,
                                             const unsigned long *strides_out, const unsigned long *strides_a,
                                             const unsigned long *strides_b, unsigned long num_dims,
                                             unsigned long total_size);

template void element_wise_inequality<double>(uint8_t *output, const double *a, const double *b, const unsigned long *dims,
                                              const unsigned long *strides_out, const unsigned long *strides_a,
                                              const unsigned long *strides_b, unsigned long num_dims,
                                              unsigned long total_size);

// CUDA kernel for element-wise negation with different strides
template <typename T>
__global__ void element_wise_negation_kernel(T *output, const T *a, const unsigned long *dims,
                                             const unsigned long *strides_out, const unsigned long *strides_a,
                                             unsigned long num_dims, unsigned long total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int flat_index_out = 0;
        int flat_index_a = 0;
        int temp = idx;
        for (int i = num_dims - 1; i >= 0; --i) {
            int dim_idx = temp % dims[i];
            flat_index_out += dim_idx * strides_out[i];
            flat_index_a += dim_idx * strides_a[i];
            temp /= dims[i];
        }
        output[flat_index_out] = -a[flat_index_a];
    }
}

template <typename T>
void element_wise_negation(T *output, const T *a, const unsigned long *dims, const unsigned long *strides_out,
                           const unsigned long *strides_a, unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    element_wise_negation_kernel<<<num_blocks, block_size>>>(output, a, dims, strides_out, strides_a, num_dims,
                                                             total_size);
}

template void element_wise_negation<float>(float *output, const float *a, const unsigned long *dims,
                                           const unsigned long *strides_out, const unsigned long *strides_a,
                                           unsigned long num_dims, unsigned long total_size);

template void element_wise_negation<double>(double *output, const double *a, const unsigned long *dims,
                                            const unsigned long *strides_out, const unsigned long *strides_a,
                                            unsigned long num_dims, unsigned long total_size);
