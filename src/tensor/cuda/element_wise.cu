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
