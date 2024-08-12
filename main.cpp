#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

auto main() -> int {
    tens<4, 4> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    for (const auto &view : A.subviews<4, 1>()) {
        tens<4> B = view;
        std::cout << B.transpose() << std::endl;
    }
}