#ifndef SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP
#define SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"

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
template <tensorial Tensor1, tensorial Tensor2> inline void blas_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    using type1 = blas_type_t<typename Tensor1::value_type>;
    using type2 = blas_type_t<typename Tensor2::value_type>;
    static_assert(std::is_same_v<type1, type2>,
                  "Tensors must have the same underlying arithmetic type for BLAS operations");
    if constexpr (fixed_tensor<Tensor1>) {
        static_assert(fixed_contiguous_tensor<Tensor1>, "Tensors must be contiguous for BLAS operations");
    }
    if constexpr (fixed_tensor<Tensor2>) {
        static_assert(fixed_contiguous_tensor<Tensor2>, "Tensors must be contiguous for BLAS operations");
    }
    if constexpr (resulting_error_checking<Tensor1::error_checking(), Tensor2::error_checking()>::value ==
                  error_checking::enabled) {
        if (!t1.is_contiguous() || !t2.is_contiguous()) {
            throw std::runtime_error("Tensors must be contiguous for BLAS operations");
        }
    }
}

// helper to check if two shapes are implicitly convertible
template <tensorial Tensor1, tensorial Tensor2>
inline void element_wise_compatible(const Tensor1 &t1, const Tensor2 &t2) {
    if constexpr (fixed_shape<typename Tensor1::shape_type> && fixed_shape<typename Tensor2::shape_type>) {
        static_assert(implicit_convertible_shapes_v<typename Tensor1::shape_type, typename Tensor2::shape_type>,
                      "Shapes must be compatible for element-wise operations");
    }
    if constexpr (Tensor1::error_checking() == error_checking::enabled ||
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
        constexpr std::size_t m = shape1[0];
        constexpr std::size_t n = shape1[1];
        constexpr std::size_t p = shape2.size() == 1 ? 1 : shape2[1];
        static_assert(implicit_convertible_shapes_v<typename Tensor1::shape_type, shape<m, n>> &&
                          implicit_convertible_shapes_v<typename Tensor2::shape_type, shape<n, p>>,
                      "Incompatible shapes for matrix multiplication");
    }
    if constexpr (resulting_error_checking<Tensor1::error_checking(), Tensor2::error_checking()>::value ==
                  error_checking::enabled) {
        std::size_t m = t1.shape()[0];
        std::size_t n = t1.shape()[1];
        std::size_t p = t2.shape().size() == 1 ? 1 : t2.shape()[1];
        if (!implicit_convertible_shapes_vector(t1.shape(), {m, n}) ||
            !implicit_convertible_shapes_vector(t2.shape(), {n, p})) {
            throw std::runtime_error("Incompatible shapes for matrix multiplication");
        }
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_OP_COMPATIBILITY_HPP