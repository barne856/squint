#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "squint/tensor/cuda/scalar.hpp"

// CUDA kernel for scalar multiplication with different strides
template <typename T>
__global__ void scalar_multiplication_kernel(T scalar, T *output, const T *a, const unsigned long *dims,
                                             const unsigned long *strides_out, const unsigned long *strides_a,
                                             unsigned long num_dims,
                                             unsigned long total_size) {
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
        output[flat_index_out] = scalar * a[flat_index_a];
    }
}

template <typename T>
void scalar_multiplication(T scalar, T *output, const T *a, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           unsigned long num_dims, unsigned long total_size) {
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    scalar_multiplication_kernel<<<num_blocks, block_size>>>(scalar, output, a, dims, strides_out, strides_a,
                                                             num_dims, total_size);
}

template void scalar_multiplication<float>(float scalar, float *output, const float *a, const unsigned long *dims,
                                           const unsigned long *strides_out, const unsigned long *strides_a,
                                           unsigned long num_dims, unsigned long total_size);

template void scalar_multiplication<double>(double scalar, double *output, const double *a, const unsigned long *dims,
                                            const unsigned long *strides_out, const unsigned long *strides_a,
                                            unsigned long num_dims, unsigned long total_size);

