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
#include "squint/core/memory.hpp"
#include "squint/quantity/quantity_math.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/math_utils.hpp"
#include "squint/util/sequence_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
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

/**
 * @brief Computes the cross product of two 3D vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The cross product of a and b.
 * @throws std::invalid_argument if the vectors are not 3D.
 */
template <tensorial T1, tensorial T2> auto cross(const T1 &a, const T2 &b) {
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        static_assert(T1::shape_type::size() == 1 && T2::shape_type::size() == 1 &&
                          std::get<0>(make_array(typename T1::shape_type{})) == 3 &&
                          std::get<0>(make_array(typename T2::shape_type{})) == 3,
                      "Cross product is only defined for 3D vectors");
    } else if constexpr (T1::error_checking() == error_checking::enabled ||
                         T2::error_checking() == error_checking::enabled) {
        if (a.rank() != 1 || b.rank() != 1 || a.shape()[0] != 3 || b.shape()[0] != 3) {
            throw std::invalid_argument("Cross product is only defined for 3D vectors");
        }
    }

    using result_type = std::common_type_t<typename T1::value_type, typename T2::value_type>;
    tensor<result_type, std::index_sequence<3>> result;

    result(0) = a(1) * b(2) - a(2) * b(1);
    result(1) = a(2) * b(0) - a(0) * b(2);
    result(2) = a(0) * b(1) - a(1) * b(0);

    return result;
}

/**
 * @brief Computes the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product of a and b.
 * @throws std::invalid_argument if the vectors have different sizes.
 */
template <tensorial T1, tensorial T2> auto dot(const T1 &a, const T2 &b) {
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

    using result_type = std::common_type_t<typename T1::value_type, typename T2::value_type>;
    result_type result = 0;

    for (size_t i = 0; i < a.shape()[0]; ++i) {
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
template <tensorial T> auto trace(const T &a) {
    if constexpr (fixed_tensor<T>) {
        static_assert(T::shape_type::size() == 2 && std::get<0>(make_array(typename T::shape_type{})) ==
                                                        std::get<1>(make_array(typename T::shape_type{})),
                      "Trace is only defined for square matrices");
    } else if constexpr (T::error_checking() == error_checking::enabled) {
        if (a.rank() != 2 || a.shape()[0] != a.shape()[1]) {
            throw std::invalid_argument("Trace is only defined for square matrices");
        }
    }

    typename T::value_type result = 0;

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
template <tensorial T> auto norm(const T &a) {
    using value_type = typename T::value_type;
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
template <tensorial T> auto squared_norm(const T &a) {
    using value_type = typename T::value_type;
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
template <tensorial T> auto normalize(const T &a) { return a / norm(a); }

/**
 * @brief Computes the mean of all elements in the tensor.
 * @param a The input tensor.
 * @return The mean value of all elements.
 */
template <tensorial T> auto mean(const T &a) {
    typename T::value_type sum = 0;
    size_t count = 0;

    for (const auto &val : a) {
        sum += val;
        ++count;
    }

    return sum / count;
}

/**
 * @brief Computes the sum of all elements in the tensor.
 * @param a The input tensor.
 * @return The sum of all elements.
 */
template <tensorial T> auto sum(const T &a) { return std::accumulate(a.begin(), a.end(), typename T::value_type(0)); }

/**
 * @brief Finds the minimum element in the tensor.
 * @param a The input tensor.
 * @return The minimum element.
 */
template <tensorial T> auto min(const T &a) { return *std::min_element(a.begin(), a.end()); }

/**
 * @brief Finds the maximum element in the tensor.
 * @param a The input tensor.
 * @return The maximum element.
 */
template <tensorial T> auto max(const T &a) { return *std::max_element(a.begin(), a.end()); }

/**
 * @brief Checks if two tensors are approximately equal within a given tolerance.
 * @param a The first tensor.
 * @param b The second tensor.
 * @param tol The tolerance for comparison (default is machine epsilon).
 * @return True if the tensors are approximately equal, false otherwise.
 */
template <tensorial T1, tensorial T2>
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

template <dynamic_tensor Tensor1, dynamic_tensor Tensor2>
auto contract(const Tensor1 &A, const Tensor2 &B, const std::vector<std::pair<size_t, size_t>> &contraction_pairs) {
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
    using result_value_type = std::common_type_t<typename Tensor1::value_type, typename Tensor2::value_type>;
    using tensor_type = tensor<result_value_type, typename Tensor1::shape_type, typename Tensor1::strides_type,
                               Tensor1::error_checking(), ownership_type::owner, memory_space::host>;

    // print permutation
    auto A_permuted = tensor_type(A.permute(A_permutation));
    auto B_permuted = tensor_type(B.permute(B_permutation));

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
    for (size_t idx : A_free_indices) {
        result_shape.push_back(A_shape[idx]);
    }
    for (size_t idx : B_free_indices) {
        result_shape.push_back(B_shape[idx]);
    }

    // Reshape result to final tensor shape
    return tensor_type(result_matrix.reshape(result_shape));
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_MATH_HPP
