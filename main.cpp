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

using namespace squint;

int main() {
    auto A = mat4x2_t<units::length>::random();
    auto b = mat4x2::random();

    std::cout << b / A << std::endl;

    auto Ad = tens_t<units::length>::random({2, 2});
    auto bd = tens::random({4, 2});

    std::cout << bd / Ad << std::endl;

    return 0;
}