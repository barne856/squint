#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

auto main() -> int {
    tens<4, 4> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tens<4, 4> B{};

    // B.subview<4>(0,0) = A.subview<4>(0,0);

    // copy rows of A to cols of B using iterators
    auto A_rows = A.rows().begin();
    auto B_cols = B.cols().begin();
    std::size_t num_rows = 4;
    for (std::size_t i = 0; i < num_rows; ++i) {
        *B_cols++ = (*A_rows++).transpose();
    }

    auto C = B;
    auto D(A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
    std::cout << D << std::endl;
}