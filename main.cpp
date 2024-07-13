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
    auto A =
        squint::fixed_tensor<double, squint::layout::column_major, squint::error_checking::disabled, 2, 3>::random();

    // std::cout << A << std::endl;
    // std::cout << A.transpose() << std::endl;
    // std::cout << (A.transpose() * A).inv() << std::endl;
    // std::cout << A.pinv() << std::endl;
    // std::cout << A.inv() << std::endl;
    // std::cout << A * A.inv() << std::endl;
    // std::cout << A * A.pinv() << std::endl;

    return 0;
}