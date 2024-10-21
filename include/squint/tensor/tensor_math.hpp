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
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/quantity/quantity_math.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/math_utils.hpp"
#include "squint/util/sequence_utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

/**
 * @brief Solves a system of linear equations.
 * @param A The matrix of coefficients.
 * @param B The right-hand side of the equations.
 * @return The pivot indices.
 * @throws std::runtime_error if the system is singular or an error occurs during the solution.
 */
template <host_tensor T1, host_tensor T2> auto solve(T1 &A, T2 &B) {
    blas_compatible(A, B);
    solve_compatible(A, B);
    static_assert(dimensionless_scalar<typename T1::value_type>);
    using blas_type = std::remove_const_t<
        std::common_type_t<blas_type_t<typename T1::value_type>, blas_type_t<typename T2::value_type>>>;

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
template <host_tensor T1, host_tensor T2> auto solve_general(T1 &A, T2 &B) {
    blas_compatible(A, B);
    solve_general_compatible(A, B);
    static_assert(dimensionless_scalar<typename T1::value_type>);
    using blas_type = std::remove_const_t<
        std::common_type_t<blas_type_t<typename T1::value_type>, blas_type_t<typename T2::value_type>>>;

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
template <host_tensor T> auto inv(const T &A) {
    inversion_compatible(A);
    static_assert(dimensionless_scalar<typename T::value_type>);
    using blas_type = blas_type_t<std::remove_const_t<typename T::value_type>>;
    using result_type =
        tensor<std::remove_const_t<typename T::value_type>, typename T::shape_type, typename T::strides_type,
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
    static_assert(host_tensor<T>, "Pseudoinverse is only supported for host tensors");
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
    static_assert(host_tensor<T>, "Pseudoinverse is only supported for host tensors");
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

/**
 * @brief Computes the determinant of a square matrix.
 * @tparam T The tensor type.
 * @param A The input matrix.
 * @return The determinant of the matrix.
 * @throws std::runtime_error if the matrix is not square or if an error occurs during computation.
 */
template <host_tensor T> auto det(const T &A) {
    blas_compatible(A, A);
    static_assert(dimensionless_scalar<typename T::value_type>);
    using result_type = std::remove_const_t<typename T::value_type>;
    if constexpr (fixed_tensor<T>) {
        constexpr auto shape = make_array(typename T::shape_type{});
        static_assert(shape.size() == 2 && shape[0] == shape[1], "Determinant is only defined for square matrices");
    } else if constexpr (T::error_checking() == error_checking::enabled) {
        if (A.rank() != 2 || A.shape()[0] != A.shape()[1]) {
            throw std::runtime_error("Determinant is only defined for square matrices");
        }
    }

    const auto n = static_cast<BLAS_INT>(A.shape()[0]);

    if (n == 0) {
        return result_type{1}; // Determinant of 0x0 matrix is 1 by convention
    }
    if (n == 1) {
        return A(0, 0);
    }
    if (n == 2) {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    if (n == 3) {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    }

    // For larger matrices, use LAPACK
    using blas_type = blas_type_t<std::remove_const_t<typename T::value_type>>;
    static_assert(std::is_same_v<blas_type, float> || std::is_same_v<blas_type, double>,
                  "Determinant is only supported for float and double types");

    // Determine matrix layout
    int layout = (A.strides()[0] == 1) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;

    // Determine leading dimension
    BLAS_INT lda = compute_leading_dimension_lapack(layout, A);

    // Create a copy of A to work with
    auto A_copy = A.copy();

    std::vector<BLAS_INT> ipiv(n);
    BLAS_INT info = 0;

    // Perform LU factorization
    // NOLINTBEGIN
    if constexpr (std::is_same_v<blas_type, float>) {
        info = LAPACKE_sgetrf(layout, n, n, reinterpret_cast<float *>(A_copy.data()), lda, ipiv.data());
    } else if constexpr (std::is_same_v<blas_type, double>) {
        info = LAPACKE_dgetrf(layout, n, n, reinterpret_cast<double *>(A_copy.data()), lda, ipiv.data());
    }
    // NOLINTEND

    if (info != 0) {
        // if LU factorization fails, det is zero
        return result_type{0};
    }

    // Compute determinant from LU factorization
    result_type det = 1;
    for (BLAS_INT i = 0; i < n; ++i) {
        if (ipiv[i] != i + 1) {
            det = -det;
        }
        det *= A_copy(i, i);
    }

    return det;
}

/**
 * @brief Compute the cross product of two 3D vectors.
 *
 * @param a The first input vector
 * @param b The second input vector
 * @param result The output vector to store the cross product result
 */
template <host_tensor T1, host_tensor T2, host_tensor T3> void cross(const T1 &a, const T2 &b, T3 &result) {
    cross_compatible(a);
    cross_compatible(b);
    cross_compatible(result);
    check_contiguous(a);
    check_contiguous(b);
    check_contiguous(result);
    blas_compatible(a, b);
    blas_compatible(b, result);

    using result_value_type = std::remove_const_t<decltype(std::declval<typename T1::value_type>() *
                                                           std::declval<typename T2::value_type>())>;
    static_assert(std::is_same_v<typename T3::value_type, result_value_type>,
                  "Result tensor must have the value type that is the product of a and b");

    result(0) = a(1) * b(2) - a(2) * b(1);
    result(1) = a(2) * b(0) - a(0) * b(2);
    result(2) = a(0) * b(1) - a(1) * b(0);
}

/**
 * @brief Computes the cross product of two 3D vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The cross product of a and b.
 * @throws std::invalid_argument if the vectors are not 3D.
 */
template <host_tensor T1, host_tensor T2> auto cross(const T1 &a, const T2 &b) {
    using result_value_type = std::remove_const_t<decltype(std::declval<typename T1::value_type>() *
                                                           std::declval<typename T2::value_type>())>;
    tensor<result_value_type, shape<3>> result;
    cross(a, b, result);
    return result;
}

/**
 * @brief Computes the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of a and b.
 * @throws std::invalid_argument if the vectors have different sizes.
 */
template <host_tensor T1, host_tensor T2> auto dot(const T1 &a, const T2 &b) {
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        static_assert(T1::shape_type::size() == 1 && T2::shape_type::size() == 1 &&
                          std::get<0>(make_array(typename T1::shape_type{})) ==
                              std::get<0>(make_array(typename T2::shape_type{})),
                      "Dot product requires vectors of the same size");
    } else if constexpr (T1::error_checking() == error_checking::enabled ||
                         T2::error_checking() == error_checking::enabled) {
        if (a.rank() != 1 || b.rank() != 1 || a.shape()[0] != b.shape()[0]) {
            throw std::invalid_argument("Dot product requires vectors of the same size");
        }
    }

    using result_type = std::remove_const_t<decltype(std::declval<typename T1::value_type>() *
                                                     std::declval<typename T2::value_type>())>;
    auto result = result_type(0);

    for (size_t i = 0; i < a.size(); ++i) {
        result += a(i) * b(i);
    }

    return result;
}

/**
 * @brief Computes the trace of a square matrix.
 * @param a The input matrix.
 * @return The trace of the matrix.
 * @throws std::invalid_argument if the matrix is not square.
 */
template <host_tensor T> auto trace(const T &a) {
    if constexpr (fixed_tensor<T>) {
        static_assert(T::shape_type::size() == 2 && std::get<0>(make_array(typename T::shape_type{})) ==
                                                        std::get<1>(make_array(typename T::shape_type{})),
                      "Trace is only defined for square matrices");
    } else if constexpr (T::error_checking() == error_checking::enabled) {
        if (a.rank() != 2 || a.shape()[0] != a.shape()[1]) {
            throw std::invalid_argument("Trace is only defined for square matrices");
        }
    }

    std::remove_const_t<typename T::value_type> result = 0;

    for (size_t i = 0; i < a.shape()[0]; ++i) {
        result += a(i, i);
    }

    return result;
}

/**
 * @brief Computes the Euclidean norm (L2 norm) of a vector.
 * @param a The input vector.
 * @return The Euclidean norm of the vector.
 */
template <host_tensor T> auto norm(const T &a) {
    using value_type = std::remove_const_t<typename T::value_type>;
    if constexpr (quantitative<value_type>) {
        return sqrt(squared_norm(a));
    } else {
        return std::sqrt(squared_norm(a));
    }
}

/**
 * @brief Computes the squared Euclidean norm of a vector.
 * @param a The input vector.
 * @return The squared Euclidean norm of the vector.
 */
template <host_tensor T> auto squared_norm(const T &a) {
    using value_type = std::remove_const_t<typename T::value_type>;
    using result_type =
        std::conditional_t<quantitative<value_type>, decltype(std::declval<value_type>() * std::declval<value_type>()),
                           value_type>;
    result_type result = result_type();

    for (const auto &val : a) {
        if constexpr (quantitative<value_type>) {
            result += pow<2>(val);
        } else {
            result += val * val;
        }
    }

    return result;
}

/**
 * @brief Normalizes a vector to have unit length.
 * @param a The input vector.
 * @return The normalized vector.
 */
template <host_tensor T> auto normalize(const T &a) { return a / norm(a); }

/**
 * @brief Computes the mean of all elements in the tensor.
 * @param a The input tensor.
 * @return The mean value of all elements.
 */
template <host_tensor T> auto mean(const T &a) { return sum(a) / a.size(); }

/**
 * @brief Computes the sum of all elements in the tensor.
 * @param a The input tensor.
 * @return The sum of all elements.
 */
template <host_tensor T> auto sum(const T &a) {
    return std::accumulate(a.begin(), a.end(), std::remove_const_t<typename T::value_type>(0));
}

/**
 * @brief Finds the minimum element in the tensor.
 * @param a The input tensor.
 * @return The minimum element.
 */
template <host_tensor T> auto min(const T &a) { return *std::min_element(a.begin(), a.end()); }

/**
 * @brief Finds the maximum element in the tensor.
 * @param a The input tensor.
 * @return The maximum element.
 */
template <host_tensor T> auto max(const T &a) { return *std::max_element(a.begin(), a.end()); }

/**
 * @brief Checks if two tensors are approximately equal within a given tolerance.
 * @param a The first tensor.
 * @param b The second tensor.
 * @param tol The tolerance for comparison (default is machine epsilon).
 * @return True if the tensors are approximately equal, false otherwise.
 */
template <host_tensor T1, host_tensor T2>
auto approx_equal(
    const T1 &a, const T2 &b,
    typename std::common_type_t<typename T1::value_type, typename T2::value_type> tol =
        std::numeric_limits<typename std::common_type_t<typename T1::value_type, typename T2::value_type>>::epsilon())
    -> bool {
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        static_assert(T1::shape_type::size() == T2::shape_type::size(),
                      "Approximate equality requires tensors of the same shape");
        static_assert(make_array(typename T1::shape_type{}) == make_array(typename T2::shape_type{}),
                      "Approximate equality requires tensors of the same shape");
    } else {
        if (a.shape() != b.shape()) {
            return false;
        }
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (!squint::approx_equal(a.data()[i], b.data()[i], tol)) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Computes the tensor product of two tensors.
 * @param A The first tensor.
 * @param B The second tensor.
 * @param contraction_pairs The pairs of indices to contract.
 * @return The tensor product of A and B.
 */
template <dynamic_tensor Tensor1, dynamic_tensor Tensor2>
auto contract(const Tensor1 &A, const Tensor2 &B, const std::vector<std::pair<size_t, size_t>> &contraction_pairs) {
    static_assert(host_tensor<Tensor1> && host_tensor<Tensor2>,
                  "Tensor contraction is only supported for host tensors");
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    size_t A_rank = A_shape.size();
    size_t B_rank = B_shape.size();

    // Determine free indices and contracted indices
    std::vector<size_t> A_free_indices;
    std::vector<size_t> A_contract_indices;
    std::vector<size_t> B_free_indices;
    std::vector<size_t> B_contract_indices;
    for (size_t i = 0; i < A_rank; ++i) {
        if (std::none_of(contraction_pairs.begin(), contraction_pairs.end(),
                         [i](const auto &pair) { return pair.first == i; })) {
            A_free_indices.push_back(i);
        } else {
            A_contract_indices.push_back(i);
        }
    }
    for (size_t i = 0; i < B_rank; ++i) {
        if (std::none_of(contraction_pairs.begin(), contraction_pairs.end(),
                         [i](const auto &pair) { return pair.second == i; })) {
            B_free_indices.push_back(i);
        } else {
            B_contract_indices.push_back(i);
        }
    }

    // Create permutation for A and B
    std::vector<size_t> A_permutation(A_free_indices);
    A_permutation.insert(A_permutation.end(), A_contract_indices.begin(), A_contract_indices.end());
    std::vector<size_t> B_permutation(B_contract_indices);
    B_permutation.insert(B_permutation.end(), B_free_indices.begin(), B_free_indices.end());

    // Permute A and B
    using result_value_type =
        std::remove_const_t<std::common_type_t<typename Tensor1::value_type, typename Tensor2::value_type>>;
    using tensor_type = tensor<result_value_type, typename Tensor1::shape_type, typename Tensor1::strides_type,
                               Tensor1::error_checking(), ownership_type::owner, memory_space::host>;

    // create permutations
    auto A_permuted = A.permute(A_permutation).copy();
    auto B_permuted = B.permute(B_permutation).copy();

    // Calculate dimensions for matrix multiplication
    size_t A_rows = std::accumulate(A_free_indices.begin(), A_free_indices.end(), 1ULL,
                                    [&A_shape](size_t acc, size_t idx) { return acc * A_shape[idx]; });
    size_t B_cols = std::accumulate(B_free_indices.begin(), B_free_indices.end(), 1ULL,
                                    [&B_shape](size_t acc, size_t idx) { return acc * B_shape[idx]; });
    size_t common_dim = std::accumulate(contraction_pairs.begin(), contraction_pairs.end(), 1ULL,
                                        [&A_shape](size_t acc, const auto &pair) { return acc * A_shape[pair.first]; });

    // Reshape permuted tensors to matrices
    auto A_matrix = A_permuted.reshape({A_rows, common_dim});
    auto B_matrix = B_permuted.reshape({common_dim, B_cols});

    // Perform matrix multiplication
    auto result_matrix = A_matrix * B_matrix;

    // Calculate result shape
    std::vector<size_t> result_shape;
    for (const size_t idx : A_free_indices) {
        result_shape.push_back(A_shape[idx]);
    }
    for (const size_t idx : B_free_indices) {
        result_shape.push_back(B_shape[idx]);
    }

    // if result shape is empty, it means the result is a scalar, so add 1
    if (result_shape.empty()) {
        result_shape.push_back(1);
    }

    // Reshape result to final tensor shape
    result_matrix.set_shape(result_shape);
    return result_matrix;
}

/**
 * @brief Helper function to determine if an index is contracted or free.
 * @tparam Sequence1 The sequence of contracted indices for the first tensor.
 * @tparam Sequence2 The sequence of contracted indices for the second tensor.
 * @tparam TensorId The tensor id (0 for the first tensor, 1 for the second tensor).
 * @tparam Idx The index to check.
 * @return True if the index is contracted, false otherwise.
 */
template <typename Sequence1, typename Sequence2, size_t TensorId, size_t Idx> struct is_contracted {
    static constexpr auto value() -> bool {
        if constexpr (TensorId == 0) {
            constexpr auto constraction_indices_a = make_array(Sequence1{});
            return !static_cast<bool>(std::none_of(constraction_indices_a.begin(), constraction_indices_a.end(),
                                                   [](auto i) { return i == Idx; }));
        } else {
            constexpr auto constraction_indices_b = make_array(Sequence2{});
            return !static_cast<bool>(std::none_of(constraction_indices_b.begin(), constraction_indices_b.end(),
                                                   [](auto i) { return i == Idx; }));
        }
    }
};

/**
 * @brief Helper function to determine if an index is contracted or free.
 * @tparam Sequence1 The sequence of contracted indices for the first tensor.
 * @tparam Sequence2 The sequence of contracted indices for the second tensor.
 * @tparam TensorId The tensor id (0 for the first tensor, 1 for the second tensor).
 * @tparam Idx The index to check.
 * @return True if the index is free, false otherwise.
 */
template <typename Sequence1, typename Sequence2, size_t TensorId, size_t Idx> struct is_free {
    static constexpr auto value() -> bool { return !is_contracted<Sequence1, Sequence2, TensorId, Idx>::value(); }
};

/**
 * @brief Helper struct to determine the types needed for tensor contraction.
 * @tparam Tensor1 The first tensor type.
 * @tparam Tensor2 The second tensor type.
 * @tparam Sequence1 The sequence of contracted indices for the first tensor.
 * @tparam Sequence2 The sequence of contracted indices for the second tensor.
 */
template <typename Tensor1, typename Tensor2, typename Sequence1, typename Sequence2> struct contraction_types {
    using A_shape = typename Tensor1::shape_type;
    using B_shape = typename Tensor2::shape_type;
    static constexpr size_t A_rank = A_shape::size();
    static constexpr size_t B_rank = B_shape::size();

    template <size_t I> using is_A_contracted = is_contracted<Sequence1, Sequence2, 0, I>;
    template <size_t I> using is_A_free = is_free<Sequence1, Sequence2, 0, I>;

    template <size_t I> using is_B_contracted = is_contracted<Sequence1, Sequence2, 1, I>;
    template <size_t I> using is_B_free = is_free<Sequence1, Sequence2, 1, I>;

    using A_free_indices = filter_sequence_t<std::make_index_sequence<A_rank>, is_A_free>;
    using A_contract_indices = filter_sequence_t<std::make_index_sequence<A_rank>, is_A_contracted>;
    using B_free_indices = filter_sequence_t<std::make_index_sequence<B_rank>, is_B_free>;
    using B_contract_indices = filter_sequence_t<std::make_index_sequence<B_rank>, is_B_contracted>;

    using A_permutation = concat_sequence_t<A_free_indices, A_contract_indices>;
    using B_permutation = concat_sequence_t<B_contract_indices, B_free_indices>;

    using result_shape =
        concat_sequence_t<select_values_t<A_shape, A_free_indices>, select_values_t<B_shape, B_free_indices>>;

    static constexpr size_t A_rows = product(select_values_t<A_shape, A_free_indices>{});
    static constexpr size_t B_cols = product(select_values_t<B_shape, B_free_indices>{});
    static constexpr size_t common_dim = product(select_values_t<A_shape, A_contract_indices>{});
};

/**
 * @brief Computes the tensor contraction of two tensors.
 * @param A The first tensor.
 * @param B The second tensor.
 * @param Sequence1 The sequence of contracted indices for the first tensor.
 * @param Sequence2 The sequence of contracted indices for the second tensor.
 * @return The result of the tensor contraction.
 */
template <fixed_tensor Tensor1, fixed_tensor Tensor2, typename Sequence1, typename Sequence2>
auto contract(const Tensor1 &A, const Tensor2 &B, const Sequence1 /*unused*/, const Sequence2 /*unused*/) {
    static_assert(host_tensor<Tensor1> && host_tensor<Tensor2>,
                  "Tensor contraction is only supported for host tensors");
    using types = contraction_types<Tensor1, Tensor2, Sequence1, Sequence2>;
    using result_value_type =
        std::remove_const_t<std::common_type_t<typename Tensor1::value_type, typename Tensor2::value_type>>;

    auto A_permuted = (A.template permute<typename types::A_permutation>()).copy();
    auto B_permuted = (B.template permute<typename types::B_permutation>()).copy();

    auto A_matrix = A_permuted.template reshape<types::A_rows, types::common_dim>();
    auto B_matrix = B_permuted.template reshape<types::common_dim, types::B_cols>();

    auto result_matrix = A_matrix * B_matrix;
    if constexpr (types::result_shape::size() != 0) {
        return (result_matrix.template reshape<typename types::result_shape>()).copy();
    } else {
        return (result_matrix.template reshape<1>()).copy();
    }
}

/**
 * @brief Computes the tensor contraction of two tensors using the Einstein summation convention.
 * @param subscripts The Einstein summation subscripts.
 * @param A The first tensor.
 * @param B The second tensor.
 * @return The result of the tensor contraction.
 *
 * The Einstein summation convention is a shorthand notation for tensor contraction. The subscripts
 * string specifies the contraction pairs and the output subscripts. For example, the subscripts "ij,jk->ik"
 * specifies the contraction of the second index of the first tensor with the first index of the second tensor,
 * and the output subscripts are the first and third indices.
 */
template <dynamic_tensor Tensor1, dynamic_tensor Tensor2>
auto einsum(const std::string &subscripts, const Tensor1 &A, const Tensor2 &B) {
    static_assert(host_tensor<Tensor1> && host_tensor<Tensor2>,
                  "Tensor contraction is only supported for host tensors");
    // Parse the subscripts
    auto pos = subscripts.find("->");
    if (pos == std::string::npos) {
        throw std::invalid_argument("Invalid einsum subscripts: missing '->'");
    }

    const std::string input_subscripts = subscripts.substr(0, pos);
    const std::string output_subscript = subscripts.substr(pos + 2);

    auto comma_pos = input_subscripts.find(',');
    if (comma_pos == std::string::npos) {
        throw std::invalid_argument("Invalid einsum subscripts: missing ','");
    }

    const std::string A_subscript = input_subscripts.substr(0, comma_pos);
    const std::string B_subscript = input_subscripts.substr(comma_pos + 1);

    // Determine contraction pairs
    std::vector<std::pair<size_t, size_t>> contraction_pairs;
    for (size_t i = 0; i < A_subscript.size(); ++i) {
        auto pos = B_subscript.find(A_subscript[i]);
        if (pos != std::string::npos && output_subscript.find(A_subscript[i]) == std::string::npos) {
            contraction_pairs.emplace_back(i, pos);
        }
    }

    // Perform contraction
    auto result = contract(A, B, contraction_pairs);

    // Determine permutation
    std::vector<size_t> permutation;
    std::string result_subscript;
    for (const char c : A_subscript) {
        if (output_subscript.find(c) != std::string::npos) {
            result_subscript += c;
        }
    }
    for (const char c : B_subscript) {
        if (output_subscript.find(c) != std::string::npos && result_subscript.find(c) == std::string::npos) {
            result_subscript += c;
        }
    }

    for (const char c : output_subscript) {
        auto pos = result_subscript.find(c);
        if (pos == std::string::npos) {
            throw std::invalid_argument("Invalid output subscript: contains indices not present in input");
        }
        permutation.push_back(pos);
    }

    // Permute result if necessary
    if (!std::ranges::empty(permutation) && !std::ranges::is_sorted(permutation)) {
        return result.permute(permutation).copy();
    }

    return result;
}

/**
 * @brief Specialization of the einsum function for a single tensor.
 * @param subscripts The Einstein summation subscripts.
 * @param tensor The input tensor.
 * @return The result of the einsum operation.
 *
 * This function is a specialization of the einsum function for a single tensor. The subscripts
 * string specifies the operation to perform on the tensor. For example, the subscripts "ij->ji"
 * specifies a matrix transpose operation.
 */
template <dynamic_tensor Tensor> auto einsum(const std::string &subscripts, const Tensor &tensor) {
    static_assert(host_tensor<Tensor>, "Tensor contraction is only supported for host tensors");
    // Parse the subscripts
    auto pos = subscripts.find("->");
    if (pos == std::string::npos) {
        throw std::invalid_argument("Invalid einsum subscripts: missing '->'");
    }

    const std::string input_subscripts = subscripts.substr(0, pos);
    const std::string output_subscripts = subscripts.substr(pos + 2);

    if (input_subscripts == output_subscripts) {
        // No operation needed
        return tensor;
    }
    if (output_subscripts.empty()) {
        // Trace operation
        using result_value_type = std::remove_const_t<typename Tensor::value_type>;
        using result_type = ::squint::tensor<result_value_type, dynamic, dynamic>;
        return result_type({1}, trace(tensor));
    }
    if (output_subscripts.size() < input_subscripts.size()) {
        // Diagonal operation
        return tensor.diag_view().copy();
    }
    // Permutation
    std::vector<size_t> permutation;
    for (const char c : output_subscripts) {
        permutation.push_back(input_subscripts.find(c));
    }
    return tensor.permute(permutation).copy();
}

// Define constexpr aliases for indices
constexpr size_t I = 0, J = 1, K = 2, L = 3, M = 4, N = 5, O = 6, P = 7, Q = 8, R = 9;

// Helper metafunction to check if a value is in a sequence
template <std::size_t Val, typename Seq> struct is_in_sequence;

template <std::size_t Val, std::size_t... Seq>
struct is_in_sequence<Val, std::index_sequence<Seq...>> : std::bool_constant<((Val == Seq) || ...)> {};

// Helper metafunction to get contraction indices
template <typename ASubscripts, typename BSubscripts> struct get_contraction_indices {
    template <std::size_t... Is> static constexpr auto helper(std::index_sequence<Is...> /*unused*/) {
        constexpr std::size_t size = sizeof...(Is);
        std::array<std::size_t, size> indices{};
        std::size_t count = 0;
        constexpr auto a_subscripts_arr = make_array(ASubscripts{});
        ((is_in_sequence<a_subscripts_arr[Is], BSubscripts>::value ? indices[count++] = Is : 0), ...);
        return std::pair{indices, count};
    }

    static constexpr auto indices_and_count = helper(std::make_index_sequence<ASubscripts::size()>{});

    template <std::size_t... Is> static constexpr auto to_sequence(std::index_sequence<Is...> /*unused*/) {
        return std::index_sequence<indices_and_count.first[Is]...>{};
    }

    using type = decltype(to_sequence(std::make_index_sequence<indices_and_count.second>{}));
};

/**
 * @brief Einsum for two fixed tensors.
 * @tparam ASubscripts The subscripts for the first tensor.
 * @tparam BSubscripts The subscripts for the second tensor.
 * @tparam OutputSubscripts The subscripts for the output tensor.
 * @tparam Tensor1 The first tensor type.
 * @tparam Tensor2 The second tensor type.
 * @param A The first tensor.
 * @param B The second tensor.
 * @return The result of the einsum operation.
 */
template <typename ASubscripts, typename BSubscripts, typename OutputSubscripts, fixed_tensor Tensor1,
          fixed_tensor Tensor2>
auto einsum(const Tensor1 &A, const Tensor2 &B) {
    static_assert(host_tensor<Tensor1> && host_tensor<Tensor2>,
                  "Tensor contraction is only supported for host tensors");
    using a_contractions = typename get_contraction_indices<ASubscripts, BSubscripts>::type;
    using b_contractions = typename get_contraction_indices<BSubscripts, ASubscripts>::type;

    auto result = contract(A, B, a_contractions{}, b_contractions{});

    // Permute result if necessary
    if constexpr (OutputSubscripts::size() > 0) {
        constexpr auto permutation = make_array(OutputSubscripts{});
        if constexpr (!std::is_sorted(permutation.begin(), permutation.end())) {
            return result.template permute<OutputSubscripts>().copy();
        } else {
            return result;
        }
    } else {
        return result;
    }
}

/**
 * @brief Einsum for a single fixed tensor.
 * @tparam Subscripts The subscripts for the tensor.
 * @tparam Tensor The tensor type.
 * @param tensor The tensor.
 * @return The result of the einsum operation.
 */
template <typename InputSubscripts, typename OutputSubscripts, typename Tensor> auto einsum(const Tensor &tensor) {
    static_assert(host_tensor<Tensor>, "Tensor contraction is only supported for host tensors");
    if constexpr (std::is_same_v<InputSubscripts, OutputSubscripts>) {
        // No operation needed
        return tensor;
    } else if constexpr (OutputSubscripts::size() == 0) {
        // Trace operation
        using value_type = std::remove_const_t<typename Tensor::value_type>;
        using result_type = ::squint::tensor<value_type, shape<1>, seq<1>>;
        return result_type{trace(tensor)};
    } else if constexpr (OutputSubscripts::size() < InputSubscripts::size()) {
        // Diagonal operation
        return tensor.diag_view().copy();
    } else {
        // Permutation
        return tensor.template permute<OutputSubscripts>().copy();
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_MATH_HPP
