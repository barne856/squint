/**
 * @file tensor_assignment.hpp
 * @brief Implementation of tensor class assignment operators.
 *
 * This file contains the implementations of assignment operators for the tensor class,
 * including assignment from other tensors and copy assignment.
 */
#ifndef SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP
#define SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/sequence_utils.hpp"

#include <stdexcept>

namespace squint {

// Assignment operator from another tensor
/**
 * @brief Assigns the contents of another tensor to this tensor.
 * @param other The tensor to assign from.
 * @return Reference to the modified tensor.
 * @throws std::runtime_error if tensors have different sizes (when error checking is enabled).
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides, ownership_type OtherOwnershipType>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator=(
    const tensor<U, OtherShape, OtherStrides, ErrorChecking, OtherOwnershipType, MemorySpace> &other) -> tensor & {
    if constexpr (fixed_shape<Shape>) {
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
    } else if constexpr (ErrorChecking == error_checking::enabled) {
        if (!implicit_convertible_shapes_vector(other.shape(), shape())) {
            throw std::runtime_error("Invalid shape conversion");
        }
    }
    auto other_begin = other.begin();
    for (auto &element : *this) {
        element = *other_begin++;
    }
    return *this;
}

// Copy assignment operator
/**
 * @brief Copies the contents of another tensor of the same type to this tensor.
 * @param other The tensor to copy from.
 * @return Reference to the modified tensor.
 * @throws std::runtime_error if tensors have different sizes (when error checking is enabled).
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator=(const tensor &other) -> tensor & {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (!implicit_convertible_shapes_vector(other.shape(), shape())) {
            throw std::runtime_error("Invalid shape conversion");
        }
    }
    auto other_begin = other.begin();
    for (auto &element : *this) {
        element = *other_begin++;
    }
    return *this;
}
} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_ASSIGNMENT_HPP