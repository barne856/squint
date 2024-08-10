#include "squint/tensor/tensor_ops.hpp"
#include "squint/tensor/tensor_types.hpp"
#include "squint/util/array_utils.hpp"
#include <iostream>
#include <utility>

using namespace squint;

int main() {
    tens<2> A{1, 2};
    tens<2, 2> B{1, 2, 3, 4};
    tens<2, 2, 2> C{1, 2, 3, 4, 5, 6, 7, 8};
    tens<2, 2, 2, 2> D{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D << std::endl;
}