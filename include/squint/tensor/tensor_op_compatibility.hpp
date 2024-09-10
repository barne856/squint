/**
 * @file tensor_op_compatibility.hpp
 * @brief Helper functions for checking tensor compatibility.
 *
 * This file contains helper functions for checking tensor compatibility
 * for various tensor operations, including element-wise operations and matrix multiplication.
 */
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
#include <utility>
#include <vector>

namespace squint {

/**
 * @brief Helper function to check if two shapes are implicitly convertible.
 * @tparam Shape1 The first shape type.
 * @tparam Shape2 The second shape type.
 * @return True if the shapes are implicitly convertible, false otherwise.
 */
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

/**
 * @brief Helper function to check if a tensor is compatible with a subview shape.
 * @tparam T The tensor type.
 * @tparam SubviewShape The subview shape type.
 * @return True if the tensor is compatible with the subview shape, false otherwise.
 */
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

/**
 * @brief Helper struct to get the underlying arithmetic type of a scalar.
 * @tparam S The scalar type.
 */
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

template <tensorial T> auto constexpr check_blas_layout(const T &t) -> void {
    if constexpr (fixed_tensor<T>) {
        // rank must be <= 2
        static_assert(make_array(typename T::shape_type{}).size() <= 2, "rank() <= 2 for t1");
        // if rank == 2, strides must start or end with 1
        if constexpr (make_array(typename T::shape_type{}).size() == 2) {
            static_assert(make_array(typename T::strides_type{})[0] == 1 ||
                              make_array(typename T::strides_type{})[1] == 1,
                          "t1 must be either row-major or column-major");
        }
    } else if constexpr (T::error_checking() == error_checking::enabled) {
        if (t.rank() > 2) {
            throw std::runtime_error("rank() <= 2 for t1");
        }
        if (t.rank() == 2) {
            if (t.strides()[0] != 1 && t.strides()[1] != 1) {
                throw std::runtime_error("t1 must be either row-major or column-major");
            }
        }
    }
}

/**
 * @brief Checks if tensors are BLAS compatible (same underlying arithmetic type).
 * @tparam Tensor1 First tensor type.
 * @tparam Tensor2 Second tensor type.
 * @param t1 First tensor.
 * @param t2 Second tensor.
 */
template <tensorial Tensor1, tensorial Tensor2> constexpr void blas_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    using type1 = blas_type_t<typename Tensor1::value_type>;
    using type2 = blas_type_t<typename Tensor2::value_type>;
    static_assert(std::is_same_v<type1, type2>,
                  "Tensors must have the same underlying arithmetic type for BLAS operations");
    check_blas_layout(t1);
    check_blas_layout(t2);
}

/**
 * @brief Checks if two tensors are compatible for element-wise operations.
 * @tparam Tensor1 First tensor type.
 * @tparam Tensor2 Second tensor type.
 * @param t1 First tensor.
 * @param t2 Second tensor.
 * @throws std::runtime_error if tensors are incompatible (when error checking is enabled).
 */
template <tensorial Tensor1, tensorial Tensor2>
constexpr void element_wise_compatible(const Tensor1 &t1, const Tensor2 &t2) {
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

template <tensorial T1, tensorial T2> constexpr void layouts_compatible(const T1 &t1, const T2 &t2) {
    // false if both ranks are 2 and layouts are different
    // 1D layouts are always compatible
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        constexpr int layout1 = make_array(typename T1::strides_type{})[0] == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
        constexpr int layout2 = make_array(typename T2::strides_type{})[0] == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
        static_assert(make_array(typename T1::shape_type{}).size() == 1 ||
                          make_array(typename T2::shape_type{}).size() == 1 || layout1 == layout2,
                      "Tensors must have the same layout for LAPACK operations");
    } else if constexpr (T1::error_checking() == error_checking::enabled ||
                         T2::error_checking() == error_checking::enabled) {
        int layout1 = t1.strides()[0] == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
        int layout2 = t2.strides()[0] == 1 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
        if (t1.rank() == 2 && t2.rank() == 2 && layout1 != layout2) {
            throw std::runtime_error("Tensors must have the same layout for LAPACK operations");
        }
    }
}

/**
 * @brief Checks if two tensors are compatible for matrix multiplication.
 * @tparam Tensor1 First tensor type.
 * @tparam Tensor2 Second tensor type.
 * @param t1 First tensor.
 * @param t2 Second tensor.
 * @throws std::runtime_error if tensors are incompatible (when error checking is enabled).
 */
template <tensorial Tensor1, tensorial Tensor2> void matrix_multiply_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    if constexpr (fixed_shape<typename Tensor1::shape_type> && fixed_shape<typename Tensor2::shape_type>) {
        constexpr auto shape1 = make_array(typename Tensor1::shape_type{});
        constexpr auto shape2 = make_array(typename Tensor2::shape_type{});

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

/**
 * @brief Helper struct to determine the resulting shape of matrix multiplication.
 * @tparam Sequence1 Shape of the first tensor.
 * @tparam Sequence2 Shape of the second tensor.
 */
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

template <typename SequenceB, typename SequenceA> struct matrix_division_sequence {
    static_assert(fixed_shape<SequenceB> || dynamic_shape<SequenceB>,
                  "SequenceB must satisfy fixed_shape or dynamic_shape concept");
    static_assert(fixed_shape<SequenceA> || dynamic_shape<SequenceA>,
                  "SequenceA must satisfy fixed_shape or dynamic_shape concept");

    template <typename B, typename A> static auto helper() {
        if constexpr (fixed_shape<B> && fixed_shape<A>) {
            constexpr auto shape_b = make_array(B{});
            constexpr auto shape_a = make_array(A{});

            constexpr auto m = shape_a[0];
            constexpr auto n = shape_a.size() == 1 ? 1 : shape_a[1];
            constexpr auto p = shape_b.size() == 1 ? 1 : shape_b[1];

            static_assert(shape_b[0] == m, "Incompatible shapes for matrix division");

            if constexpr (p == 1) {
                return std::index_sequence<n>{};
            } else {
                return std::index_sequence<n, p>{};
            }
        } else {
            return std::vector<std::size_t>{}; // Placeholder, actual computation done at runtime
        }
    }

    using type = decltype(helper<SequenceB, SequenceA>());
};

template <typename SequenceB, typename SequenceA>
using matrix_division_sequence_t = typename matrix_division_sequence<SequenceB, SequenceA>::type;

/**
 * @brief Computes the leading dimension for BLAS operations.
 * @tparam Tensor1 Tensor type.
 * @param op Transpose operation.
 * @param t Tensor.
 * @return Leading dimension for BLAS operations.
 */
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

/**
 * @brief Computes the leading dimension for LAPACK operations.
 * @tparam Tensor1 Tensor type.
 * @param layout Matrix layout (row-major or column-major).
 * @param t Tensor.
 * @return Leading dimension for LAPACK operations.
 */
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

template <tensorial T1> void check_contiguous(const T1 &t1) {
    if constexpr (fixed_tensor<T1>) {
        static_assert(fixed_contiguous_tensor<T1>, "tensor must be contiguous");
    } else if (T1::error_checking() == error_checking::enabled) {
        if (!t1.is_contiguous()) {
            throw std::runtime_error("tensor must be contiguous");
        }
    }
}

/**
 * @brief Checks if two tensors are compatible for cross product operations.
 * @tparam T The tensor type.
 * @param a The tensor to check.
 * @throws std::invalid_argument if the tensor is not 3D (when error checking is enabled).
 */
template <host_tensor T> constexpr void cross_compatible(const T &a) {
    if constexpr (fixed_tensor<T>) {
        static_assert(T::shape_type::size() == 1, "Cross product is only supported for 1D tensors");
        static_assert(std::get<0>(make_array(typename T::shape_type{})) == 3,
                      "Cross product is only supported for 3D vectors");
    } else if (T::error_checking() == error_checking::enabled) {
        if (a.rank() != 1 || a.shape()[0] != 3) {
            throw std::invalid_argument("Cross product is only supported for 3D vectors");
        }
    }
}

/**
 * @brief Checks if two tensors are compatible for solve operations.
 * @tparam T1 First tensor type.
 * @tparam T2 Second tensor type.
 * @param A First tensor.
 * @param B Second tensor.
 * @throws std::runtime_error if tensors are incompatible (when error checking is enabled).
 */
template <tensorial T1, tensorial T2> auto solve_compatible(const T1 &A, const T2 &B) {
    using error_type = resulting_error_checking<T1::error_checking(), T2::error_checking()>;
    layouts_compatible(A, B);
    check_contiguous(A);
    check_contiguous(B);
    // check for compatible shapes and strides
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        // check shapes at compile time
        static_assert(make_array(typename T1::shape_type{})[0] == make_array(typename T1::shape_type{})[1],
                      "A must be square");
        static_assert(make_array(typename T1::shape_type{})[0] == make_array(typename T2::shape_type{})[0],
                      "A and B must have the same number of rows");
    } else if (error_type::value == error_checking::enabled) {
        // check shapes at runtime
        if (A.shape()[0] != A.shape()[1]) {
            throw std::runtime_error("A must be square");
        }
        if (A.shape()[0] != B.shape()[0]) {
            throw std::runtime_error("A and B must have the same number of rows");
        }
    }
}

/**
 * @brief Checks if two tensors are compatible for general solve operations.
 * @tparam T1 First tensor type.
 * @tparam T2 Second tensor type.
 * @param A First tensor.
 * @param B Second tensor.
 * @throws std::runtime_error if tensors are incompatible (when error checking is enabled).
 */
template <tensorial T1, tensorial T2> auto solve_general_compatible(const T1 &A, const T2 &B) {
    using error_type = resulting_error_checking<T1::error_checking(), T2::error_checking()>;
    layouts_compatible(A, B);
    // Check for compatible shapes
    if constexpr (fixed_tensor<T1> && fixed_tensor<T2>) {
        // Check shapes at compile time
        if constexpr (make_array(typename T1::shape_type{}).size() == 1) {
            static_assert(make_array(typename T2::shape_type{})[0] >= make_array(typename T1::shape_type{})[0],
                          "B must have enough rows to hold the input");
        } else {
            static_assert(
                make_array(typename T2::shape_type{})[0] >=
                    std::max(make_array(typename T1::shape_type{})[0], make_array(typename T1::shape_type{})[1]),
                "B must have enough rows to hold the result");
        }
    } else if (error_type::value == error_checking::enabled) {
        // Check shapes at runtime
        if (A.rank() == 1) {
            if (B.shape()[0] < A.shape()[0]) {
                throw std::runtime_error("B must have enough rows to hold the input");
            }
        } else {
            if (B.shape()[0] < std::max(A.shape()[0], A.shape()[1])) {
                throw std::runtime_error("B must have enough rows to hold the result");
            }
        }
    }
}

/**
 * @brief Checks if a tensor is compatible for inversion.
 * @tparam T The tensor type.
 * @param t The tensor to check.
 * @throws std::runtime_error if the tensor is not square or not 2D (when error checking is enabled).
 */
template <tensorial T> constexpr void inversion_compatible(const T &t) {
    if constexpr (fixed_tensor<T>) {
        static_assert(T::shape_type::size() == 2, "Tensor must be 2D for inversion");
        static_assert(std::get<0>(make_array(typename T::shape_type{})) ==
                          std::get<1>(make_array(typename T::shape_type{})),
                      "Tensor must be square for inversion");
    } else if constexpr (T::error_checking() == error_checking::enabled) {
        if (t.rank() != 2) {
            throw std::runtime_error("Tensor must be 2D for inversion");
        }
        if (t.shape()[0] != t.shape()[1]) {
            throw std::runtime_error("Tensor must be square for inversion");
        }
    }
    check_contiguous(t);
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP