/**
 * @file tensor_constructors.hpp
 * @brief Implementation of tensor class constructors.
 *
 * This file contains the implementations of various constructors for the tensor class,
 * including default construction, initialization from lists or arrays, and construction
 * of owning and non-owning tensors with fixed or dynamic shapes.
 *
 */

#ifndef SQUINT_TENSOR_TENSOR_CONSTRUCTORS_HPP
#define SQUINT_TENSOR_TENSOR_CONSTRUCTORS_HPP

#include "squint/tensor/tensor.hpp"
#include <algorithm>
#include <stdexcept>

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(std::initializer_list<T> init)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (init.size() != this->size()) {
            throw std::invalid_argument("Initializer list size does not match tensor size");
        }
    }
    std::copy(init.begin(), init.end(), data_.begin());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(const T &value)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
    : data_() {
    std::fill(data_.begin(), data_.end(), value);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(
    const std::array<T, product(Shape{})> &elements)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
    : data_(elements) {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (elements.size() != product(Shape{})) {
            throw std::invalid_argument("Input array size does not match tensor size");
        }
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(Shape shape, Strides strides)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(std::move(strides)) {
    data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(Shape shape, layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(compute_strides(l)) {
    data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(std::vector<size_t> shape,
                                                                             const std::vector<T> &elements, layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(compute_strides(l)), data_(elements) {
    if constexpr (ErrorChecking == error_checking::enabled) {
        size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        if (elements.size() != total_size) {
            throw std::invalid_argument("Input vector size does not match tensor size");
        }
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(std::vector<size_t> shape, const T &value,
                                                                             layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(compute_strides(l)) {
    size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
    data_.resize(total_size, value);
}

// construct from another tensor of a different shape (allows for implicit conversion to tensor of same shape with
// trailing 1's removed) allows implicit conversion for compatible tensors of various shape of the same ownership type
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(
    const tensor<U, OtherShape, OtherStrides, ErrorChecking, OwnershipType, MemorySpace> &other)
    requires fixed_shape<Shape>
{
    if constexpr (OwnershipType == ownership_type::owner) {
        // for owner ownership, only shape must be convertible
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
        std::copy(other.begin(), other.end(), begin());
    } else {
        // for reference ownership, both strides and shape must be convertible
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
        static_assert(implicit_convertible_strides_v<Strides, OtherStrides>, "Invalid strides conversion");
        data_ = other.data();
    }
}

// Allows implicit conversion for compatible tensors of various shape from reference ownership type to owner ownership
// type
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename U, typename OtherShape, typename OtherStrides>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(
    const tensor<U, OtherShape, OtherStrides, ErrorChecking, ownership_type::reference, MemorySpace> &other)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (dynamic_shape<Shape>) {
        static_assert(dynamic_shape<OtherShape>, "Invalid shape conversion");
        static_assert(dynamic_shape<OtherStrides>, "Invalid strides conversion");
        shape_ = other.shape();
        strides_ = other.strides();
        data_.resize(other.size());
        std::copy(other.begin(), other.end(), begin());
    } else {
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
        std::copy(other.begin(), other.end(), begin());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(T *data, Shape shape, Strides strides)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::reference)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(T *data)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::reference)
    : data_(data) {}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_CONSTRUCTORS_HPP