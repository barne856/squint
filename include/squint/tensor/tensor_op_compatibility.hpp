#ifndef SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP
#define SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/tensor/blas_backend.hpp"
#include "squint/util/sequence_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <utility>

namespace squint {
// helper to check if two shapes are implicitly convertible with vectors
inline auto implicit_convertible_shapes_vector(std::vector<std::size_t> shape1,
                                               std::vector<std::size_t> shape2) -> bool {
    auto size1 = shape1.size();
    auto size2 = shape2.size();
    auto min_size = std::min(size1, size2);
    // Check if the common elements are the same
    for (std::size_t i = 0; i < min_size; ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    // Check if the extra elements in the longer sequence are all 1's
    if (size1 > size2) {
        for (std::size_t i = min_size; i < size1; ++i) {
            if (shape1[i] != 1) {
                return false;
            }
        }
    } else if (size2 > size1) {
        for (std::size_t i = min_size; i < size2; ++i) {
            if (shape2[i] != 1) {
                return false;
            }
        }
    }
    return true;
}

// Helper function to check if tensor dimensions are divisible by subview dimensions
template <fixed_tensor T, typename SubviewShape> constexpr auto subview_compatible() -> bool {
    constexpr auto shape_arr = make_array(typename T::shape_type{});
    constexpr auto subview_arr = make_array(SubviewShape{});
    std::size_t min_length = shape_arr.size() < subview_arr.size() ? shape_arr.size() : subview_arr.size();
    for (std::size_t i = 0; i < min_length; ++i) {
        if (shape_arr[i] % subview_arr[i] != 0) {
            return false;
        }
    }
    return true;
}

// helper to get underlying arithmetic type of a scalar
template <scalar S> struct blas_type {
    // helper to get the underlying arithmetic type of a scalar
    static auto helper() {
        if constexpr (quantitative<S>) {
            return typename S::value_type{};
        } else {
            return S{};
        }
    }
    using type = decltype(helper());
};

// alias for the underlying arithmetic type of a tensor
template <scalar S> using blas_type_t = typename blas_type<S>::type;

// helper to check if tensors are blas compatible, i.e. have the same underlying arithmetic type
template <tensorial Tensor1, tensorial Tensor2>
inline void blas_compatible(const Tensor1 & /*t1*/, const Tensor2 & /*t2*/) {
    using type1 = blas_type_t<typename Tensor1::value_type>;
    using type2 = blas_type_t<typename Tensor2::value_type>;
    static_assert(std::is_same_v<type1, type2>,
                  "Tensors must have the same underlying arithmetic type for BLAS operations");
}

// helper to check if two shapes are implicitly convertible
template <tensorial Tensor1, tensorial Tensor2>
inline void element_wise_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    if constexpr (fixed_shape<typename Tensor1::shape_type> && fixed_shape<typename Tensor2::shape_type>) {
        static_assert(implicit_convertible_shapes_v<typename Tensor1::shape_type, typename Tensor2::shape_type>,
                      "Shapes must be compatible for element-wise operations");
    } else if constexpr (Tensor1::error_checking() == error_checking::enabled ||
                         Tensor2::error_checking() == error_checking::enabled) {
        if (!implicit_convertible_shapes_vector(t1.shape(), t2.shape())) {
            throw std::runtime_error("Shapes must be compatible for element-wise operations");
        }
    }
}

// helper to check if two shapes are compatible for matrix multiplication
template <tensorial Tensor1, tensorial Tensor2>
inline void matrix_multiply_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    if constexpr (fixed_shape<typename Tensor1::shape_type> && fixed_shape<typename Tensor2::shape_type>) {
        constexpr auto shape1 = make_array(typename Tensor1::shape_type{});
        constexpr auto shape2 = make_array(typename Tensor2::shape_type{});
        static_assert(shape1.size() == 1 || shape1.size() == 2, "Invalid shape for tensor 1");
        static_assert(shape2.size() == 1 || shape2.size() == 2, "Invalid shape for tensor 2");
        constexpr auto stride1 = make_array(typename Tensor1::strides_type{});
        constexpr auto stride2 = make_array(typename Tensor2::strides_type{});
        // both strides must start or end with 1
        static_assert((stride1[0] == 1 || stride1[shape1.size() - 1] == 1) &&
                          (stride2[0] == 1 || stride2[shape2.size() - 1] == 1),
                      "Invalid strides for matrix multiplication");

        if constexpr (shape1.size() == 1) {
            // Vector-matrix multiplication
            constexpr std::size_t m = shape1[0];
            constexpr std::size_t n = shape2[0];
            constexpr std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
            static_assert(m == p, "Incompatible shapes for vector-matrix multiplication");
            static_assert(implicit_convertible_shapes_v<typename Tensor1::shape_type, shape<m>> &&
                              implicit_convertible_shapes_v<typename Tensor2::shape_type, shape<n, p>>,
                          "Incompatible shapes for vector-matrix multiplication");
        } else {
            // Matrix-matrix multiplication
            constexpr std::size_t m = shape1[0];
            constexpr std::size_t n = shape1[1];
            constexpr std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
            static_assert(shape1[1] == shape2[0], "Incompatible shapes for matrix multiplication");
            static_assert(implicit_convertible_shapes_v<typename Tensor1::shape_type, shape<m, n>> &&
                              implicit_convertible_shapes_v<typename Tensor2::shape_type, shape<n, p>>,
                          "Incompatible shapes for matrix multiplication");
        }
    } else if constexpr (resulting_error_checking<Tensor1::error_checking(), Tensor2::error_checking()>::value ==
                         error_checking::enabled) {
        auto shape1 = t1.shape();
        auto shape2 = t2.shape();

        if (shape1.size() != 1 && shape1.size() != 2) {
            throw std::runtime_error("Invalid shape for tensor 1");
        }
        if (shape2.size() != 1 && shape2.size() != 2) {
            throw std::runtime_error("Invalid shape for tensor 2");
        }

        auto stride1 = t1.strides();
        auto stride2 = t2.strides();

        // both strides must start or end with 1
        if ((stride1[0] != 1 && stride1[shape1.size() - 1] != 1) ||
            (stride2[0] != 1 && stride2[shape2.size() - 1] != 1)) {
            throw std::runtime_error("Invalid strides for matrix multiplication");
        }

        if (shape1.size() == 1) {
            // Vector-matrix multiplication
            std::size_t m = shape1[0];
            std::size_t n = shape2[0];
            std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
            if (m != p || !implicit_convertible_shapes_vector(t1.shape(), {m}) ||
                !implicit_convertible_shapes_vector(t2.shape(), {n, p})) {
                throw std::runtime_error("Incompatible shapes for vector-matrix multiplication");
            }
        } else {
            // Matrix-matrix multiplication
            std::size_t m = shape1[0];
            std::size_t n = shape1[1];
            std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
            if (shape1[1] != shape2[0] || !implicit_convertible_shapes_vector(t1.shape(), {m, n}) ||
                !implicit_convertible_shapes_vector(t2.shape(), {n, p})) {
                throw std::runtime_error("Incompatible shapes for matrix multiplication");
            }
        }
    }
}

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

template <tensorial Tensor1> auto compute_leading_dimension_blas(CBLAS_TRANSPOSE op, const Tensor1 &t) -> BLAS_INT {
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

template <tensorial Tensor1> auto compute_leading_dimension_lapack(int layout, const Tensor1 &t) -> BLAS_INT {
    auto N = t.shape().size();
    auto num_rows = t.shape()[0];
    auto num_cols = N == 1 ? 1 : t.shape()[1];
    auto col_stride = N == 1 ? t.strides()[0] : t.strides()[1];
    auto row_stride = t.strides()[0];

    if (layout == LAPACK_ROW_MAJOR) {
        return static_cast<BLAS_INT>(num_cols * col_stride);
    } // ColumnMajor
    return static_cast<BLAS_INT>(num_rows * row_stride);
}

template <tensorial T1, tensorial T2> auto solve_compatible(const T1 &A, const T2 &B) {
    using error_type = resulting_error_checking<T1::error_checking(), T2::error_checking()>;
    // check for rank and contiguous independently
    if constexpr (fixed_tensor<T1>) {
        static_assert(fixed_contiguous_tensor<T1>, "A must be contiguous");
        static_assert(make_array(typename T1::shape_type{}).size() <= 2, "rank() <= 2 for A");
    } else if (error_type::value == error_checking::enabled) {
        if (!A.is_contiguous()) {
            throw std::runtime_error("A must be contiguous");
        }
        if (A.rank() > 2) {
            throw std::runtime_error("rank() <= 2 for A");
        }
    }
    if constexpr (fixed_tensor<T2>) {
        static_assert(fixed_contiguous_tensor<T2>, "B must be contiguous");
        static_assert(make_array(typename T2::shape_type{}).size() <= 2, "rank() <= 2 for B");
    } else if (error_type::value == error_checking::enabled) {
        if (!B.is_contiguous()) {
            throw std::runtime_error("B must be contiguous");
        }
        if (B.rank() > 2) {
            throw std::runtime_error("rank() <= 2 for B");
        }
    }
    // check for compatible shapes and strides
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        // check shapes at compile time
        static_assert(make_array(typename T1::shape_type{})[0] == make_array(typename T1::shape_type{})[1],
                      "A must be square");
        static_assert(make_array(typename T1::shape_type{})[0] == make_array(typename T2::shape_type{})[0],
                      "A and B must have the same number of rows");

        // check strides at compile time
        if constexpr (make_array(typename T2::shape_type{}).size() == 1) {
            static_assert(make_array(typename T1::strides_type{})[0] == 1 ||
                              make_array(typename T1::strides_type{})[1] == 1,
                          "A must be either row-major or column-major");
        } else {
            static_assert(
                (make_array(typename T1::strides_type{})[0] == 1 && make_array(typename T2::strides_type{})[0] == 1) ||
                    (make_array(typename T1::strides_type{})[1] == 1 &&
                     make_array(typename T2::strides_type{})[1] == 1),
                "A and B must have the same layout (both row-major or both column-major)");
        }
    } else if (error_type::value == error_checking::enabled) {
        // check shapes at runtime
        if (A.shape()[0] != A.shape()[1]) {
            throw std::runtime_error("A must be square");
        }
        if (A.shape()[0] != B.shape()[0]) {
            throw std::runtime_error("A and B must have the same number of rows");
        }

        // check strides at runtime
        if (B.rank() == 1) {
            if (A.strides()[0] != 1 && A.strides()[1] != 1) {
                throw std::runtime_error("A must be either row-major or column-major");
            }
        } else {
            if ((A.strides()[0] == 1 && B.strides()[0] != 1) || (A.strides()[1] == 1 && B.strides()[1] != 1)) {
                throw std::runtime_error("A and B must have the same layout (both row-major or both column-major)");
            }
        }
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP