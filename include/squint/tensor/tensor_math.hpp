/**
 * @file tensor_math.hpp
 * @brief Mathematical operations for tensor objects.
 *
 * This file contains implementations of mathematical operations on tensors,
 * including solving linear systems of equations.
 */
#ifndef SQUINT_TENSOR_TENSOR_MATH_HPP
#define SQUINT_TENSOR_TENSOR_MATH_HPP

#include "squint/core/concepts.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace squint {

/**
 * @brief Solves a system of linear equations.
 * @param A The matrix of coefficients.
 * @param B The right-hand side of the equations.
 * @return The pivot indices.
 * @throws std::runtime_error if the system is singular or an error occurs during the solution.
 */
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

/**
 * @brief Solves a general system of linear equations (overdetermined or underdetermined).
 * @param A The matrix of coefficients.
 * @param B The right-hand side of the equations.
 * @return The pivot indices.
 * @throws std::runtime_error if an error occurs during the solution.
 */
template <tensorial T1, tensorial T2> auto solve_general(T1 &A, T2 &B) {
    blas_compatible(A, B);
    solve_general_compatible(A, B);
    static_assert(dimensionless_scalar<typename T1::value_type>);
    using blas_type = std::common_type_t<blas_type_t<typename T1::value_type>, blas_type_t<typename T2::value_type>>;

    // Compute dimensions
    auto m = static_cast<BLAS_INT>(A.shape()[0]);
    auto n = static_cast<BLAS_INT>(A.shape()[1]);
    auto nrhs = static_cast<BLAS_INT>(B.rank() == 1 ? 1 : B.shape()[1]);

    // Determine transpose operations
    CBLAS_TRANSPOSE op_a = (A.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;
    CBLAS_TRANSPOSE op_b = (B.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;

    // Determine matrix layout
    int layout = (A.strides()[0] == 1) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // Determine leading dimensions
    BLAS_INT lda = compute_leading_dimension_lapack(layout, A);
    BLAS_INT ldb = compute_leading_dimension_lapack(layout, B);

    int info = 0;
    // NOLINTBEGIN
    if constexpr (std::is_same_v<blas_type, float>) {
        info = LAPACKE_sgels(layout, 'N', m, n, nrhs, reinterpret_cast<float *>((A.data())), lda,
                             reinterpret_cast<float *>((B.data())), ldb);
    }
    if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgels(layout, 'N', m, n, nrhs, reinterpret_cast<double *>((A.data())), lda,
                             reinterpret_cast<double *>((B.data())), ldb);
    }
    // NOLINTEND
    if (info != 0) {
        throw std::runtime_error("LAPACKE_gels error code: " + std::to_string(info));
    }

    return info;
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_MATH_HPP
