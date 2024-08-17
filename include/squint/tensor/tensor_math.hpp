#ifndef SQUINT_TENSOR_TENSOR_MATH_HPP
#define SQUINT_TENSOR_TENSOR_MATH_HPP

#include "squint/core/concepts.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace squint {

template <tensorial T1, tensorial T2> auto solve(T1 &A, T2 &B) {
    blas_compatible(A, B);
    solve_compatible(A, B);
    static_assert(dimensionless_scalar<typename T1::value_type>);
    using blas_type = std::common_type_t<blas_type_t<typename T1::value_type>, blas_type_t<typename T2::value_type>>;

    // Compute dimensions
    auto n = static_cast<BLAS_INT>(A.shape()[0]);
    auto nrhs = static_cast<BLAS_INT>(B.rank() == 1 ? 1 : B.shape()[1]);

    // Determine transpose operations
    int layout = (A.strides()[0] == 1) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // Determine leading dimensions
    BLAS_INT lda = compute_leading_dimension_lapack(layout, A);
    BLAS_INT ldb = compute_leading_dimension_lapack(layout, B);

    // print debug info
    std::cout << "layout: " << layout << std::endl;
    std::cout << "lda: " << lda << std::endl;
    std::cout << "ldb: " << ldb << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "nrhs: " << nrhs << std::endl;

    int info = 0;
    std::vector<BLAS_INT> ipiv(n);
    // NOLINTBEGIN
    if constexpr (std::is_same_v<blas_type, float>) {
        info = LAPACKE_sgesv(layout, n, nrhs, reinterpret_cast<float *>((A.data())), lda, ipiv.data(),
                             reinterpret_cast<float *>((B.data())), ldb);
    }
    if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgesv(layout, n, nrhs, reinterpret_cast<double *>((A.data())), lda, ipiv.data(),
                             reinterpret_cast<double *>((B.data())), ldb);
    }
    // NOLINTEND
    if (info != 0) {
        throw std::runtime_error("LAPACKE_gesv error code: " + std::to_string(info));
    }
    return ipiv;
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_MATH_HPP
