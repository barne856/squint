#include "squint/tensor/tensor_ops.hpp"
#include "squint/tensor/tensor_types.hpp"
#include "squint/util/array_utils.hpp"
#include <iostream>
#include <utility>

using namespace squint;

int main() {
    tens<2, 2, 2> B{1, 2, 3, 4, 5, 6, 7, 8};
    for(const auto &elem : B) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    std::cout << B << std::endl;
}