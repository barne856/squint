#ifndef SQUINT_TENSOR_TENSOR_OPS_HPP
#define SQUINT_TENSOR_TENSOR_OPS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/sequence_utils.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

template <typename Sequence1, typename Sequence2> struct matrix_multiply_sequence {
    static_assert(fixed_shape<Sequence1> || dynamic_shape<Sequence1>,
                  "Sequence1 must satisfy fixed_shape or dynamic_shape concept");
    static_assert(fixed_shape<Sequence2> || dynamic_shape<Sequence2>,
                  "Sequence2 must satisfy fixed_shape or dynamic_shape concept");

    template <typename S1, typename S2> static auto helper() {
        if constexpr (fixed_shape<S1> && fixed_shape<S2>) {
            constexpr auto arr1 = make_array(S1{});
            constexpr auto arr2 = make_array(S2{});

            if constexpr (arr1.size() == 1) {
                // Vector-matrix multiplication
                constexpr std::size_t m = arr1[0];
                constexpr std::size_t p = arr2.size() == 1 ? 1 : arr2[1];
                static_assert(m == p, "Dimensions must match for vector-matrix multiplication");
                return std::index_sequence<m, p>{};
            } else {
                // Matrix-matrix multiplication
                static_assert(arr1[1] == arr2[0], "Inner dimensions must match for matrix multiplication");
                constexpr std::size_t m = arr1[0];
                constexpr std::size_t p = arr2.size() == 1 ? 1 : arr2[1];
                return std::index_sequence<m, p>{};
            }
        } else {
            return std::vector<std::size_t>{}; // Placeholder, actual computation done at runtime
        }
    }

    using type = decltype(helper<Sequence1, Sequence2>());
};

template <typename Sequence1, typename Sequence2>
using matrix_multiply_sequence_t = typename matrix_multiply_sequence<Sequence1, Sequence2>::type;

template <tensorial Tensor1> auto compute_leading_dimension(CBLAS_TRANSPOSE op, const Tensor1 &t) -> BLAS_INT {
    if (op == CBLAS_TRANSPOSE::CblasNoTrans) {
        if (t.rank() == 1) {
            return static_cast<BLAS_INT>(t.shape()[0] * t.strides().back());
        }
        return static_cast<BLAS_INT>(t.strides().back());
    }
    if (t.rank() == 1) {
        return static_cast<BLAS_INT>(t.shape()[0] * t.strides()[0]);
    }
    return static_cast<BLAS_INT>(t.strides()[0]);
}

// General Matrix-Matrix Multiplication
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
    BLAS_INT lda = compute_leading_dimension(op_a, t1);
    BLAS_INT ldb = compute_leading_dimension(op_b, t2);
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
            cblas_sgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<float *>(const_cast<typename Tensor1::value_type *>(t1.data())), lda,
                        reinterpret_cast<float *>(const_cast<typename Tensor2::value_type *>(t2.data())), ldb, beta,
                        reinterpret_cast<float *>(result.data()), ldc);
            // NOLINTEND
        } else if constexpr (std::is_same_v<blas_type, double>) {
            // NOLINTBEGIN
            cblas_dgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<double *>(const_cast<typename Tensor1::value_type *>(t1.data())), lda,
                        reinterpret_cast<double *>(const_cast<typename Tensor2::value_type *>(t2.data())), ldb, beta,
                        reinterpret_cast<double *>(result.data()), ldc);
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
            cblas_sgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<float *>(const_cast<typename Tensor1::value_type *>(t1.data())), lda,
                        reinterpret_cast<float *>(const_cast<typename Tensor2::value_type *>(t2.data())), ldb, beta,
                        reinterpret_cast<float *>(result.data()), ldc);
            // NOLINTEND
        } else if constexpr (std::is_same_v<blas_type, double>) {
            // NOLINTBEGIN
            cblas_dgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<double *>(const_cast<typename Tensor1::value_type *>(t1.data())), lda,
                        reinterpret_cast<double *>(const_cast<typename Tensor2::value_type *>(t2.data())), ldb, beta,
                        reinterpret_cast<double *>(result.data()), ldc);
            // NOLINTEND
        }
        return result;
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OPS_HPP