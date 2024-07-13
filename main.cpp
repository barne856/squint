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
    auto A = tens({2, 2}, {2, 1, 3, 4});

    std::cout << A << std::endl;
    std::cout << A.inv() << std::endl;

    auto B = tens({2, 2}, {2, 3, 1, 4});

    std::cout << B.transpose() << std::endl;
    std::cout << B.transpose().inv() << std::endl;

    std::cout << (A.inv() == A.inv()) << std::endl;

    std::cout << approx_equal(A.inv(), A.pinv(), 0.003F) << std::endl;

    return 0;
}