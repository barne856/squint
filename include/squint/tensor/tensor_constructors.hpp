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
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (init.size() != this->size()) {
            throw std::invalid_argument("Initializer list size does not match tensor size");
        }
    }

    if constexpr (fixed_tensor<tensor>) {
        std::copy(init.begin(), init.end(), data_.begin());
    } else {
        data_ = std::vector<T>(init.begin(), init.end());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(const T &value)
    requires(OwnershipType == ownership_type::owner)
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