#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <device_launch_parameters.h>


// CUDA kernel for element-wise addition with different strides
template <typename T>
__global__ void elementWiseAdditionKernel(T* output, const T* input1, const T* input2,
                                          const int* dims, const int* stridesOut,
                                          const int* stridesIn1, const int* stridesIn2,
                                          int numDims, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize) {
        int flatIndexOut = 0;
        int flatIndexIn1 = 0;
        int flatIndexIn2 = 0;
        int temp = idx;
        for (int i = numDims - 1; i >= 0; --i) {
            int dimIdx = temp % dims[i];
            flatIndexOut += dimIdx * stridesOut[i];
            flatIndexIn1 += dimIdx * stridesIn1[i];
            flatIndexIn2 += dimIdx * stridesIn2[i];
            temp /= dims[i];
        }
        output[flatIndexOut] = input1[flatIndexIn1] + input2[flatIndexIn2];
    }
}

template <typename T>
void elementWiseAddition(std::vector<T>& output, const std::vector<T>& input1,
                         const std::vector<T>& input2, const std::vector<int>& dims,
                         const std::vector<int>& stridesOut,
                         const std::vector<int>& stridesIn1,
                         const std::vector<int>& stridesIn2) {
    int numDims = dims.size();
    int totalSize = 1;
    for (int dim : dims) {
        totalSize *= dim;
    }

    // Allocate device memory
    T *d_output, *d_input1, *d_input2;
    int *d_dims, *d_stridesOut, *d_stridesIn1, *d_stridesIn2;
    cudaMalloc(&d_output, totalSize * sizeof(T));
    cudaMalloc(&d_input1, totalSize * sizeof(T));
    cudaMalloc(&d_input2, totalSize * sizeof(T));
    cudaMalloc(&d_dims, numDims * sizeof(int));
    cudaMalloc(&d_stridesOut, numDims * sizeof(int));
    cudaMalloc(&d_stridesIn1, numDims * sizeof(int));
    cudaMalloc(&d_stridesIn2, numDims * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input1, input1.data(), totalSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), totalSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims, dims.data(), numDims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stridesOut, stridesOut.data(), numDims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stridesIn1, stridesIn1.data(), numDims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stridesIn2, stridesIn2.data(), numDims * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel using cudaLaunchKernel
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;
    void* args[] = {&d_output, &d_input1, &d_input2, &d_dims, &d_stridesOut, &d_stridesIn1, &d_stridesIn2, &numDims, &totalSize};
    
    cudaLaunchKernel((void*)elementWiseAdditionKernel<T>, dim3(numBlocks), dim3(blockSize), args, 0, nullptr);

    // Copy result back to host
    cudaMemcpy(output.data(), d_output, totalSize * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_dims);
    cudaFree(d_stridesOut);
    cudaFree(d_stridesIn1);
    cudaFree(d_stridesIn2);
}