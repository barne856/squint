#include "squint/core/concepts.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <utility>
#include <vector>

using namespace squint;

auto main() -> int {
    tens<2, 2> A{1, 2, 3, 4};
    tens<2, 2> B{1, 2, 3, 4};
    auto C = tensor<length, shape<4, 4>>::arange(length(1), length(1));

    // // inner product (row x col)
    // std::cout << B.subview<1, 2>(0, 0) * A.subview<2>(0, 0) << std::endl;
    // // outer product (col x row)
    // std::cout << A.subview<2>(0, 0) * B.subview<1, 2>(0, 0) << std::endl;
    // // matrix vector product
    // std::cout << A * B.subview<2>(0, 0) << std::endl;
    // // vector matrix product
    // std::cout << B.subview<1, 2>(0, 0) * A << std::endl;
    // // matrix matrix product
    // std::cout << A * B << std::endl;
    // std::cout << A * B.transpose() << std::endl;
    std::cout << A << std::endl;
    std::cout << C.subview<shape<2, 2>, seq<1, 2>>({0, 0}) << std::endl;
    std::cout << A * C.subview<2, 2>(0, 0) << std::endl;
}