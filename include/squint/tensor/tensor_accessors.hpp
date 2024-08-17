/**
 * @file tensor_accessors.hpp
 * @brief Implementation of tensor class accessor methods.
 *
 * This file contains the implementations of various accessor methods for the tensor class,
 * including methods to retrieve the rank, shape, strides, size, and data pointer of the tensor.
 * These methods provide essential information about the tensor's structure and contents.
 *
 */
#ifndef SQUINT_TENSOR_TENSOR_ACCESSORS_HPP
#define SQUINT_TENSOR_TENSOR_ACCESSORS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"
#include <cstddef>
#include <functional>
#include <numeric>

namespace squint {

// Rank accessor
/**
 * @brief Returns the rank (number of dimensions) of the tensor.
 * @return The rank of the tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::rank() const -> std::size_t {
    if constexpr (fixed_shape<Shape>) {
        return Shape::size();
    } else {
        return shape_.size();
    }
}

// Shape accessor
/**
 * @brief Returns the shape of the tensor.
 * @return A reference to the tensor's shape.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::shape() const -> const index_type & {
    if constexpr (fixed_shape<Shape>) {
        static const auto shape_array = make_array(Shape{});
        return shape_array;
    } else {
        return shape_;
    }
}

// Strides accessor
/**
 * @brief Returns the strides of the tensor.
 * @return A reference to the tensor's strides.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::strides() const -> const index_type & {
    if constexpr (fixed_shape<Strides>) {
        static const auto strides_array = make_array(Strides{});
        return strides_array;
    } else {
        return strides_;
    }
}

// Size accessor
/**
 * @brief Returns the total number of elements in the tensor.
 * @return The total number of elements.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::size() const -> std::size_t {
    if constexpr (fixed_shape<Shape>) {
        return product(Shape{});
    } else if constexpr (OwnershipType == ownership_type::owner) {
        return data_.size();
    } else {
        return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
    }
}

// Const data pointer accessor
/**
 * @brief Returns a const pointer to the underlying data of the tensor.
 * @return A const pointer to the tensor's data.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::data() const -> const T * {
    if constexpr (OwnershipType == ownership_type::owner) {
        return data_.data();
    } else {
        return data_;
    }
}

// Non-const data pointer accessor
/**
 * @brief Returns a non-const pointer to the underlying data of the tensor.
 * @return A pointer to the tensor's data.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::data() -> T * {
    return const_cast<T *>(std::as_const(*this).data());
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_ACCESSORS_HPP