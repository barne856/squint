#ifndef SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP
#define SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP

#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"

#include <iostream>

namespace squint {
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator=(
    const tensor<U, OtherShape, OtherStrides, ErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    if constexpr (fixed_shape<Shape>) {
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
    } else if constexpr (ErrorChecking == error_checking::enabled) {
        if (size() != other.size()) {
            throw std::runtime_error("Cannot assign tensor with different size");
        }
    }
    std::copy(other.begin(), other.end(), begin());
    return *this;
}

// Copy Assignment Operator
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator=(const tensor &other) -> tensor & {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (size() != other.size()) {
            throw std::runtime_error("Cannot assign tensor with different size");
        }
    }
    std::copy(other.begin(), other.end(), begin());
    return *this;
}
} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP