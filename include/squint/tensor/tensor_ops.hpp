/**
 * @file tensor_ops.hpp
 * @brief Tensor operations for tensor objects.
 *
 * This file contains implementations of tensor operations on tensors,
 * including matrix-matrix multiplication.
 */
#ifndef SQUINT_TENSOR_TENSOR_OPS_HPP
#define SQUINT_TENSOR_TENSOR_OPS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_math.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

/**
 * @brief General matrix-matrix multiplication operator.
 * @param t1 The first tensor to multiply.
 * @param t2 The second tensor to multiply.
 * @return A new tensor containing the result of the multiplication.
 */
template <tensorial Tensor1, tensorial Tensor2>
auto operator*(const Tensor1 &t1, const Tensor2 &t2)
    requires(host_tensor<Tensor1> && host_tensor<Tensor2>)
{
    matrix_multiply_compatible(t1, t2);
    blas_compatible(t1, t2);
    using blas_type =
        std::common_type_t<blas_type_t<typename Tensor1::value_type>, blas_type_t<typename Tensor2::value_type>>;
    using result_value_type =
        decltype(std::declval<typename Tensor1::value_type>() * std::declval<typename Tensor2::value_type>());
    using result_error_checking = resulting_error_checking<Tensor1::error_checking(), Tensor2::error_checking()>;
    using result_shape_type = matrix_multiply_sequence_t<typename Tensor1::shape_type, typename Tensor2::shape_type>;

    // Compute dimensions
    auto m = static_cast<BLAS_INT>(t1.shape()[0]);
    auto n = static_cast<BLAS_INT>(t2.rank() == 1 ? 1 : t2.shape()[1]);
    auto k = static_cast<BLAS_INT>(t1.rank() == 1 ? 1 : t1.shape()[1]);

    // Determine transpose operations
    CBLAS_TRANSPOSE op_a = (t1.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;
    CBLAS_TRANSPOSE op_b = (t2.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;

    // Compute leading dimensions
    BLAS_INT lda = compute_leading_dimension_blas(op_a, t1);
    BLAS_INT ldb = compute_leading_dimension_blas(op_b, t2);
    BLAS_INT ldc = m;

    // Scaling factors
    blas_type alpha = 1;
    blas_type beta = 0;

    if constexpr (fixed_tensor<Tensor1> && fixed_tensor<Tensor2>) {
        using strides_type = strides::column_major<result_shape_type>;
        using result_type = tensor<result_value_type, result_shape_type, strides_type, result_error_checking::value,
                                   ownership_type::owner, memory_space::host>;
        result_type result{};
        if constexpr (std::is_same_v<blas_type, float>) {
            // NOLINTBEGIN
            cblas_sgemm(
                CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                reinterpret_cast<float *>(const_cast<std::remove_const_t<typename Tensor1::value_type> *>(t1.data())),
                lda,
                reinterpret_cast<float *>(const_cast<std::remove_const_t<typename Tensor2::value_type> *>(t2.data())),
                ldb, beta, reinterpret_cast<float *>(result.data()), ldc);
            // NOLINTEND
        } else if constexpr (std::is_same_v<blas_type, double>) {
            // NOLINTBEGIN
            cblas_dgemm(
                CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                reinterpret_cast<double *>(const_cast<std::remove_const_t<typename Tensor1::value_type> *>(t1.data())),
                lda,
                reinterpret_cast<double *>(const_cast<std::remove_const_t<typename Tensor2::value_type> *>(t2.data())),
                ldb, beta,
                reinterpret_cast<double *>(
                    const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                ldc);
            // NOLINTEND
        }
        return result;
    } else {
        using strides_type = std::vector<std::size_t>;
        using result_type = tensor<result_value_type, result_shape_type, strides_type, result_error_checking::value,
                                   ownership_type::owner, memory_space::host>;
        result_type result({static_cast<std::size_t>(m), static_cast<std::size_t>(n)}, layout::column_major);
        if constexpr (std::is_same_v<blas_type, float>) {
            // NOLINTBEGIN
            cblas_sgemm(
                CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                reinterpret_cast<float *>(const_cast<std::remove_const_t<typename Tensor1::value_type> *>(t1.data())),
                lda,
                reinterpret_cast<float *>(const_cast<std::remove_const_t<typename Tensor2::value_type> *>(t2.data())),
                ldb, beta,
                reinterpret_cast<float *>(
                    const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                ldc);
            // NOLINTEND
        } else if constexpr (std::is_same_v<blas_type, double>) {
            // NOLINTBEGIN
            cblas_dgemm(
                CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                reinterpret_cast<double *>(const_cast<std::remove_const_t<typename Tensor1::value_type> *>(t1.data())),
                lda,
                reinterpret_cast<double *>(const_cast<std::remove_const_t<typename Tensor2::value_type> *>(t2.data())),
                ldb, beta,
                reinterpret_cast<double *>(
                    const_cast<std::remove_const_t<typename result_type::value_type> *>(result.data())),
                ldc);
            // NOLINTEND
        }
        return result;
    }
}

// NOLINTBEGIN
template <tensorial T1, tensorial T2> auto operator/(const T1 &B, const T2 &A) {
    using blas_type = std::common_type_t<blas_type_t<typename T1::value_type>, blas_type_t<typename T2::value_type>>;
    using result_value_type =
        decltype(std::declval<typename T1::value_type>() / std::declval<typename T2::value_type>());
    using result_error_checking = resulting_error_checking<T1::error_checking(), T2::error_checking()>;
    using result_shape_type = matrix_division_sequence_t<typename T1::shape_type, typename T2::shape_type>;
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        using strides_type = strides::column_major<result_shape_type>;
        using result_type = tensor<result_value_type, result_shape_type, strides_type, result_error_checking::value,
                                   ownership_type::owner, memory_space::host>;
        result_type X{};
        // Create mutable dimensionless copy of A
        tensor<blas_type, typename T2::shape_type, strides::column_major<typename T2::shape_type>, T2::error_checking(),
               ownership_type::owner, memory_space::host>
            A_copy;
        auto A_copy_iter = A_copy.begin();
        for (const auto &elem : A) {
            if constexpr (quantitative<typename T2::value_type>) {
                *A_copy_iter = elem.value();
            } else {
                *A_copy_iter = elem;
            }
            ++A_copy_iter;
        }
        if constexpr (make_array(result_shape_type{})[0] > make_array(typename T1::shape_type{})[0]) {
            // copy B into X casting dimension type
            if constexpr (make_array(result_shape_type{}).size() == 1) {
                auto X_iter = X.begin();
                for (const auto &elem : B) {
                    if constexpr (quantitative<typename T1::value_type>) {
                        *X_iter = result_value_type(elem.value());
                    } else {
                        *X_iter = result_value_type(elem);
                    }
                    ++X_iter;
                }
            } else {
                auto X_view = X.template subview<make_array(typename T1::shape_type{})[0],
                                                 make_array(typename T1::shape_type{})[1]>(0, 0);
                auto X_iter = X_view.begin();
                for (const auto &elem : B) {
                    if constexpr (quantitative<typename T1::value_type>) {
                        *X_iter = result_value_type(elem.value());
                    } else {
                        *X_iter = result_value_type(elem);
                    }
                    ++X_iter;
                }
            }
            solve_general(A_copy, X);
        } else {
            // create mutable copy of B
            tensor<std::remove_const_t<typename T1::value_type>, typename T1::shape_type,
                   strides::column_major<typename T1::shape_type>, T1::error_checking(), ownership_type::owner,
                   memory_space::host>
                B_copy{B};
            solve_general(A_copy, B_copy);
            // copy B into X casting dimension type
            if constexpr (make_array(result_shape_type{}).size() == 1) {
                auto B_copy_iter = B_copy.begin();
                for (auto &elem : X) {
                    if constexpr (quantitative<typename T1::value_type>) {
                        elem = result_value_type((*B_copy_iter).value());
                    } else {
                        elem = result_value_type(*B_copy_iter);
                    }
                    ++B_copy_iter;
                }
            } else {
                auto B_view =
                    B_copy.template subview<make_array(result_shape_type{})[0], make_array(result_shape_type{})[1]>(0,
                                                                                                                    0);
                auto B_copy_iter = B_view.begin();
                for (auto &elem : X) {
                    if constexpr (quantitative<typename T1::value_type>) {
                        elem = result_value_type((*B_copy_iter).value());
                    } else {
                        elem = result_value_type(*B_copy_iter);
                    }
                    ++B_copy_iter;
                }
            }
        }
        return X;
    } else {
        using result_type = tensor<result_value_type, result_shape_type, std::vector<std::size_t>,
                                   result_error_checking::value, ownership_type::owner, memory_space::host>;
        auto n = A.shape().size() == 1 ? 1 : A.shape()[1];
        auto p = B.shape().size() == 1 ? 1 : B.shape()[1];
        result_type X({n, p}, layout::column_major);
        // Create mutable dimensionless copy of A
        tensor<blas_type, std::vector<std::size_t>, std::vector<std::size_t>, T2::error_checking(),
               ownership_type::owner, memory_space::host>
            A_copy(A.shape());
        auto A_copy_iter = A_copy.begin();
        for (const auto &elem : A) {
            if constexpr (quantitative<typename T2::value_type>) {
                *A_copy_iter = elem.value();
            } else {
                *A_copy_iter = elem;
            }
            ++A_copy_iter;
        }
        if (X.shape()[0] > B.shape()[0]) {
            // copy B into X casting dimension type
            auto start_indices = std::vector<std::size_t>(B.shape().size(), 0);
            auto X_view = X.subview(B.shape(), start_indices);
            auto X_iter = X_view.begin();
            for (const auto &elem : B) {
                if constexpr (quantitative<typename T1::value_type>) {
                    *X_iter = result_value_type(elem.value());
                } else {
                    *X_iter = result_value_type(elem);
                }
                ++X_iter;
            }
            solve_general(A_copy, X);
        } else {
            // create mutable copy of B
            tensor<std::remove_const_t<typename T1::value_type>, std::vector<std::size_t>, std::vector<std::size_t>,
                   T1::error_checking(), ownership_type::owner, memory_space::host>
                B_copy{B};
            solve_general(A_copy, B_copy);
            // copy B into X casting dimension type
            auto start_indices = std::vector<std::size_t>(X.shape().size(), 0);
            auto B_view = B_copy.subview(X.shape(), start_indices);
            auto B_iter = B_view.begin();
            for (auto &elem : X) {
                if constexpr (quantitative<typename T1::value_type>) {
                    elem = result_value_type((*B_iter).value());
                } else {
                    elem = result_value_type(*B_iter);
                }
                ++B_iter;
            }
        }
        return X;
    }
}
// NOLINTEND

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OPS_HPP