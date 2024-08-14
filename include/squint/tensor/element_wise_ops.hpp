#ifndef SQUINT_TENSOR_ELEMENT_WISE_OPS_HPP
#define SQUINT_TENSOR_ELEMENT_WISE_OPS_HPP

#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator+=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    element_wise_compatible(*this, other);
    std::transform(begin(), end(), other.begin(), begin(), std::plus{});
    return *this;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator-=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    element_wise_compatible(*this, other);
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
    element_wise_compatible(*this, other);
    return std::equal(begin(), end(), other.begin());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
          enum ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator!=(
    const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other) const
    -> bool {
    element_wise_compatible(*this, other);
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
    element_wise_compatible(lhs, rhs);
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
    element_wise_compatible(lhs, rhs);
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