#ifndef SQUINT_TENSOR_TENSOR_OPS_HPP
#define SQUINT_TENSOR_TENSOR_OPS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/sequence_utils.hpp"
#include <type_traits>

#include <iostream>

namespace squint {

// Helper to determine the resulting sequence of a matrix multiplication
template <typename Sequence1, typename Sequence2> struct matrix_multiply_sequence {
    static_assert(fixed_shape<Sequence1> || dynamic_shape<Sequence1>,
                  "Sequence1 must satisfy fixed_shape or dynamic_shape concept");
    static_assert(fixed_shape<Sequence2> || dynamic_shape<Sequence2>,
                  "Sequence2 must satisfy fixed_shape or dynamic_shape concept");

    template <typename S1, typename S2> static auto helper() {
        if constexpr (fixed_shape<S1> && fixed_shape<S2>) {
            constexpr auto arr1 = make_array(S1{});
            constexpr auto arr2 = make_array(S2{});
            static_assert(arr1[1] == arr2[0], "Inner dimensions must match for matrix multiplication");
            constexpr std::size_t m = arr1[0];
            constexpr std::size_t n = arr1[1];
            constexpr std::size_t p = arr2.size() == 1 ? 1 : arr2[1];
            return std::index_sequence<m, p>{};
        } else {
            return std::vector<std::size_t>{}; // Placeholder, actual computation done at runtime
        }
    }

    using type = decltype(helper<Sequence1, Sequence2>());
};

// Alias template for matrix_multiply_sequence
template <typename Sequence1, typename Sequence2>
using matrix_multiply_sequence_t = typename matrix_multiply_sequence<Sequence1, Sequence2>::type;

// Function to compute matrix multiplication result for dynamic shapes
template <tensorial Tensor1, tensorial Tensor2>
auto matrix_multiply_shapes(const Tensor1 &t1, const Tensor2 &t2) -> std::vector<std::size_t> {
    auto shape1 = t1.shape();
    auto shape2 = t2.shape();
    std::size_t m = shape1[0];
    std::size_t n = shape1[1];
    std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
    return {m, p};
}

// General Matrix-Matrix Multiplication
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, typename U, typename OtherShape, typename OtherStrides,
          error_checking OtherErrorChecking, ownership_type OtherOwnershipType>
auto operator*(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t1,
               const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &t2) {
    matrix_multiply_compatible(t1, t2);
    blas_compatible<tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>,
                    tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace>>(t1, t2);
    using blas_type = std::common_type_t<blas_type_t<T>, blas_type_t<U>>;
    using result_value_type = decltype(std::declval<T>() * std::declval<U>());
    using result_error_checking = resulting_error_checking<ErrorChecking, OtherErrorChecking>;
    using result_shape_type = matrix_multiply_sequence_t<Shape, OtherShape>;
    using strides_type = std::conditional_t<fixed_shape<result_shape_type>, strides::row_major<result_shape_type>,
                                            std::vector<std::size_t>>;
    using result_type = tensor<result_value_type, result_shape_type, strides_type, result_error_checking::value,
                               ownership_type::owner, MemorySpace>;

    // Compute dimensions
    BLAS_INT m = static_cast<BLAS_INT>(t1.shape()[0]);
    BLAS_INT n = static_cast<BLAS_INT>(t2.rank() == 1 ? 1 : t2.shape()[1]);
    BLAS_INT k = static_cast<BLAS_INT>(t1.shape()[1]);

    // Determine transpose operations
    CBLAS_TRANSPOSE op_a = (t1.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;
    CBLAS_TRANSPOSE op_b = (t2.strides()[0] == 1) ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;

    // Compute leading dimensions
    BLAS_INT lda = static_cast<BLAS_INT>((op_a == CBLAS_TRANSPOSE::CblasNoTrans) ? t1.shape()[0] : t1.strides()[0]);
    BLAS_INT ldb = static_cast<BLAS_INT>((op_b == CBLAS_TRANSPOSE::CblasNoTrans) ? t2.shape()[0] : t2.strides()[0]);
    BLAS_INT ldc = 2;

    // print debug info
    std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
    std::cout << "lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc << std::endl;
    std::cout << "op_a: " << (op_a == CBLAS_TRANSPOSE::CblasNoTrans ? "CblasNoTrans" : "CblasTrans") << std::endl;
    std::cout << "op_b: " << (op_b == CBLAS_TRANSPOSE::CblasNoTrans ? "CblasNoTrans" : "CblasTrans") << std::endl;

    // Scaling factors
    blas_type alpha = 1;
    blas_type beta = 0;

    if constexpr (fixed_tensor<result_type>) {
        result_type result;
        if constexpr (std::is_same_v<blas_type, float>) {
            cblas_sgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<float *>(const_cast<T *>(t1.data())), lda,
                        reinterpret_cast<float *>(const_cast<U *>(t2.data())), ldb, beta,
                        reinterpret_cast<float *>(result.data()), ldc);
        } else if constexpr (std::is_same_v<blas_type, double>) {
            cblas_dgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<double *>(const_cast<T *>(t1.data())), lda,
                        reinterpret_cast<double *>(const_cast<U *>(t2.data())), ldb, beta,
                        reinterpret_cast<double *>(result.data()), ldc);
        }
        return result;
    } else {
        result_type result(matrix_multiply_shapes(t1, t2));
        if constexpr (std::is_same_v<blas_type, float>) {
            cblas_sgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<float *>(const_cast<T *>(t1.data())), lda,
                        reinterpret_cast<float *>(const_cast<U *>(t2.data())), ldb, beta,
                        reinterpret_cast<float *>(result.data()), ldc);
        } else if constexpr (std::is_same_v<blas_type, double>) {
            cblas_dgemm(CBLAS_ORDER::CblasColMajor, op_a, op_b, m, n, k, alpha,
                        reinterpret_cast<double *>(const_cast<T *>(t1.data())), lda,
                        reinterpret_cast<double *>(const_cast<U *>(t2.data())), ldb, beta,
                        reinterpret_cast<double *>(result.data()), ldc);
        }
        return result;
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OPS_HPP