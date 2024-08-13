#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

auto main() -> int {
    tens<4, 4> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tens<4, 4> B(tens<1, 4>{1, 2, 3, 4}, tens<1, 4>{5, 6, 7, 8}, tens<1, 4>{9, 10, 11, 12}, tens<1, 4>{13, 14, 15, 16});

    std::cout << B << std::endl;
    std::cout << B.subview<std::index_sequence<2, 2>, std::index_sequence<2, 2>>({0, 0}) << std::endl;

    std::cout << B.size() << std::endl;
    std::cout << B.rank() << std::endl;
    std::cout << B.shape()[0] << std::endl;
}