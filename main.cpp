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
    mat3x4 A{{1, 1, 1, 1, 2, 3, 4, 5, 3, 5, 2, 4}};
    vec3 b{{10, 20, 30}};

    std::cout << A.transpose() * (A * A.transpose()).inv() * b << std::endl;

    return 0;
}