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
#include "squint/core/memory.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/sequence_utils.hpp"

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
        info = LAPACKE_sgesv(
            layout, n, nrhs,
            reinterpret_cast<float *>(const_cast<std::remove_const_t<typename T1::value_type> *>(A.data())), lda,
            ipiv.data(),
            reinterpret_cast<float *>(const_cast<std::remove_const_t<typename T2::value_type> *>(B.data())), ldb);
    }
    if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgesv(
            layout, n, nrhs,
            reinterpret_cast<double *>(const_cast<std::remove_const_t<typename T1::value_type> *>(A.data())), lda,
            ipiv.data(),
            reinterpret_cast<double *>(const_cast<std::remove_const_t<typename T2::value_type> *>(B.data())), ldb);
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
        info = LAPACKE_sgels(
            layout, 'N', m, n, nrhs,
            reinterpret_cast<float *>(const_cast<std::remove_const_t<typename T1::value_type> *>(A.data())), lda,
            reinterpret_cast<float *>(const_cast<std::remove_const_t<typename T2::value_type> *>(B.data())), ldb);
    }
    if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgels(
            layout, 'N', m, n, nrhs,
            reinterpret_cast<double *>(const_cast<std::remove_const_t<typename T1::value_type> *>(A.data())), lda,
            reinterpret_cast<double *>(const_cast<std::remove_const_t<typename T2::value_type> *>(B.data())), ldb);
    }
    // NOLINTEND
    if (info != 0) {
        throw std::runtime_error("LAPACKE_gels error code: " + std::to_string(info));
    }

    return info;
}

/**
 * @brief Computes the inverse of a square matrix.
 * @param A The matrix to invert.
 * @return The inverted matrix.
 * @throws std::runtime_error if the matrix is singular or an error occurs during inversion.
 */
template <tensorial T> auto inv(const T &A) {
    inversion_compatible(A);
    static_assert(dimensionless_scalar<typename T::value_type>);
    using blas_type = blas_type_t<typename T::value_type>;
    using result_type = tensor<typename T::value_type, typename T::shape_type, typename T::strides_type,
                               T::error_checking(), ownership_type::owner, memory_space::host>;

    // Create a copy of A to work with
    result_type result = A;

    // Compute dimensions
    auto n = static_cast<BLAS_INT>(A.shape()[0]);

    // Determine matrix layout
    int layout = (A.strides()[0] == 1) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // Determine leading dimension
    BLAS_INT lda = compute_leading_dimension_lapack(layout, A);

    std::vector<BLAS_INT> ipiv(n);
    int info = 0;

    // Perform LU factorization
    // NOLINTBEGIN
    if constexpr (std::is_same_v<blas_type, float>) {
        info = LAPACKE_sgetrf(layout, n, n,
                              reinterpret_cast<float *>(
                                  const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                              lda, ipiv.data());
    } else if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgetrf(layout, n, n,
                              reinterpret_cast<double *>(
                                  const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                              lda, ipiv.data());
    }

    if (info != 0) {
        throw std::runtime_error("LU factorization failed: " + std::to_string(info));
    }

    // Compute the inverse
    if constexpr (std::is_same_v<blas_type, float>) {
        info = LAPACKE_sgetri(layout, n,
                              reinterpret_cast<float *>(
                                  const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                              lda, ipiv.data());
    } else if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgetri(layout, n,
                              reinterpret_cast<double *>(
                                  const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                              lda, ipiv.data());
    }
    // NOLINTEND

    if (info != 0) {
        throw std::runtime_error("Matrix inversion failed: " + std::to_string(info));
    }

    return result;
}

/**
 * @brief Computes the Moore-Penrose pseudoinverse of a matrix.
 *
 * This function calculates the pseudoinverse of a matrix using the following formulas:
 * - For overdetermined or square systems (m >= n): pinv(A) = (A^T * A)^-1 * A^T
 * - For underdetermined systems (m < n): pinv(A) = A^T * (A * A^T)^-1
 *
 * The pseudoinverse is a generalization of the inverse matrix and can be used for matrices
 * that are not square or not of full rank.
 *
 * @tparam T The tensor type of the input matrix.
 * @param A The input matrix as a tensor.
 * @return The pseudoinverse of A as a tensor of the same type as the input.
 *
 */
template <fixed_tensor T> auto pinv(const T &A) {
    constexpr int m = make_array(typename T::shape_type{})[0];
    constexpr int n = make_array(typename T::shape_type{}).size() > 1 ? make_array(typename T::shape_type{})[1] : 1;

    if constexpr (m >= n) {
        // Overdetermined or square system: pinv(A) = (A^T * A)^-1 * A^T
        auto AtA = A.transpose() * A;
        return inv(AtA) * A.transpose();
    } else {
        // Underdetermined system: pinv(A) = A^T * (A * A^T)^-1
        auto AAt = A * A.transpose();
        return A.transpose() * inv(AAt);
    }
}

template <dynamic_tensor T> auto pinv(const T &A) {
    int m = A.shape()[0];
    int n = A.rank() > 1 ? A.shape()[1] : 1;

    if (m >= n) {
        // Overdetermined or square system: pinv(A) = (A^T * A)^-1 * A^T
        auto AtA = A.transpose() * A;
        return inv(AtA) * A.transpose();
    }
    // Underdetermined system: pinv(A) = A^T * (A * A^T)^-1
    auto AAt = A * A.transpose();
    return A.transpose() * inv(AAt);
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_MATH_HPP
