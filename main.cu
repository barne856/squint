#include "element_wise_addition.cuh"
#include <iostream>

int main() {
    // Example usage for 3D arrays with different strides
    std::vector<int> dims = {2, 3, 4};
    std::vector<int> stridesOut = {12, 4, 1};  // C-style contiguous
    std::vector<int> stridesIn1 = {12, 4, 1};  // C-style contiguous
    std::vector<int> stridesIn2 = {1, 2, 6};   // Fortran-style contiguous
    int totalSize = 2 * 3 * 4;

    std::vector<float> input1(totalSize);
    std::vector<float> input2(totalSize);
    std::vector<float> output(totalSize);

    // Initialize input arrays
    for (int i = 0; i < totalSize; ++i) {
        input1[i] = static_cast<float>(i);
        input2[i] = static_cast<float>(i * 2);
    }

    // Perform element-wise addition
    elementWiseAddition(output, input1, input2, dims, stridesOut, stridesIn1, stridesIn2);

    // Print results
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                int idx = i * stridesOut[0] + j * stridesOut[1] + k * stridesOut[2];
                std::cout << output[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}