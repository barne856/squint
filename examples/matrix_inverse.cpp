#include "squint/tensor/tensor_math.hpp"
#include <cassert>
#include <iostream>
#include <squint/geometry.hpp>
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>

using namespace squint;

auto main() -> int {
    try {
        const mat3 A{1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 4.0F, 5.0F, 6.0F, 0.0F};
        std::cout << A << '\n';
        auto A_inv = inv(A);
        std::cout << A_inv << '\n';
    } catch (const std::exception &e) {
        std::cerr << "Matrix inversion failed with error: " << e.what() << '\n';
    }
    return 0;
}