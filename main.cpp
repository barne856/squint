#include "squint/dynamic_tensor.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/linear_algebra.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor_base.hpp"
#include <chrono>
#include <concepts>
#include <iostream>
#include <type_traits>
#ifdef BLAS_BACKEND_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

int main() {
    auto m = squint::mat2x4({1, 1, 2, 1, 3, 1, 4, 1});
    auto b = squint::mat4x2({10, 5, 0, 0, 10, 5, 0, 0});

    std::cout << "m = " << m << std::endl;
    std::cout << "b = " << b << std::endl;

    squint::solve_lls(m, b);

    // std::cout << "x = " << b.subview<2>(squint::slice{0, 2}) << std::endl;
    std::cout << "x = " << b << std::endl;

    return 0;
}