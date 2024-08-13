#ifndef SQUINT_TENSOR_ELEMENT_WISE_OPS_HPP
#define SQUINT_TENSOR_ELEMENT_WISE_OPS_HPP

#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
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

// helper to check if two shapes are implicitly convertible
template <tensorial Tensor1, tensorial Tensor2> inline void shapes_convertible(const Tensor1 &t1, const Tensor2 &t2) {
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

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator+=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    shapes_convertible(*this, other);
    std::transform(begin(), end(), other.begin(), begin(), std::plus{});
    return *this;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator-=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    shapes_convertible(*this, other);
    std::transform(begin(), end(), other.begin(), begin(), std::minus{});
    return *this;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator==(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) const
    -> bool {
    shapes_convertible(*this, other);
    return std::equal(begin(), end(), other.begin());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator!=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) const
    -> bool {
    shapes_convertible(*this, other);
    return !std::equal(begin(), end(), other.begin());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator-() const -> tensor {
    tensor result(*this);
    std::transform(result.begin(), result.end(), result.begin(), std::negate{});
    return result;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, typename U, typename OtherShape, typename OtherStrides,
          enum error_checking OtherErrorChecking, enum ownership_type OtherOwnershipType>
auto operator+(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &lhs,
               const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &rhs)
    -> tensor<decltype(std::declval<T>() + std::declval<U>()),
              std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Shape, std::vector<std::size_t>>,
              std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Strides, std::vector<std::size_t>>,
              resulting_error_checking<ErrorChecking, OtherErrorChecking>::value, ownership_type::owner, MemorySpace> {
    shapes_convertible(lhs, rhs);
    tensor<decltype(std::declval<T>() + std::declval<U>()),
           std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Shape, std::vector<std::size_t>>,
           std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Strides, std::vector<std::size_t>>,
           resulting_error_checking<ErrorChecking, OtherErrorChecking>::value, ownership_type::owner, MemorySpace>
        result(lhs);
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::plus{});
    return result;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, typename U, typename OtherShape, typename OtherStrides,
          enum error_checking OtherErrorChecking, enum ownership_type OtherOwnershipType>
auto operator-(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &lhs,
               const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &rhs)
    -> tensor<decltype(std::declval<T>() - std::declval<U>()),
              std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Shape, std::vector<std::size_t>>,
              std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Strides, std::vector<std::size_t>>,
              resulting_error_checking<ErrorChecking, OtherErrorChecking>::value, ownership_type::owner, MemorySpace> {
    shapes_convertible(lhs, rhs);
    tensor<decltype(std::declval<T>() - std::declval<U>()),
           std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Shape, std::vector<std::size_t>>,
           std::conditional_t<fixed_shape<Shape> && fixed_shape<OtherShape>, Strides, std::vector<std::size_t>>,
           resulting_error_checking<ErrorChecking, OtherErrorChecking>::value, ownership_type::owner, MemorySpace>
        result(lhs);
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::minus{});
    return result;
}

} // namespace squint

#endif // SQUINT_TENSOR_ELEMENT_WISE_OPS_HPP