/**
 * @file tensor_shape_manipulations.hpp
 * @brief Implementation of tensor shape manipulation methods.
 *
 * This file contains the implementations of various methods for manipulating the shape
 * of tensors, including reshaping, flattening, and transposing. These methods are
 * implemented for both fixed and dynamic shapes, and for const and non-const tensors.
 *
 */

#ifndef SQUINT_TENSOR_TENSOR_SHAPE_MANIPULATIONS_HPP
#define SQUINT_TENSOR_TENSOR_SHAPE_MANIPULATIONS_HPP

#include "squint/tensor/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <size_t... NewDims>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::reshape()
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
{
    constexpr size_t new_size = (NewDims * ...);
    constexpr size_t current_size = product(Shape{});
    static_assert(new_size == current_size, "New shape must have the same number of elements as the original tensor");

    using NewShape = std::index_sequence<NewDims...>;
    using NewStrides = typename std::conditional_t<std::is_same_v<Strides, column_major_strides<Shape>>,
                                                   column_major_strides<NewShape>, row_major_strides<NewShape>>;

    return tensor<T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <size_t... NewDims>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::reshape() const
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
{
    constexpr size_t new_size = (NewDims * ...);
    constexpr size_t current_size = product(Shape{});
    static_assert(new_size == current_size, "New shape must have the same number of elements as the original tensor");

    using NewShape = std::index_sequence<NewDims...>;
    using NewStrides = typename std::conditional_t<std::is_same_v<Strides, column_major_strides<Shape>>,
                                                   column_major_strides<NewShape>, row_major_strides<NewShape>>;

    return tensor<const T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::flatten() {
    if constexpr (fixed_shape<Shape>) {
        constexpr size_t total_size = product(Shape{});
        using FlatShape = std::index_sequence<total_size>;
        using FlatStrides = std::index_sequence<1>;

        return tensor<T, FlatShape, FlatStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    } else {
        std::vector<size_t> flat_shape = {this->size()};
        std::vector<size_t> flat_strides = {1};

        return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), flat_shape, flat_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::flatten() const {
    if constexpr (fixed_shape<Shape>) {
        constexpr size_t total_size = product(Shape{});
        using FlatShape = std::index_sequence<total_size>;
        using FlatStrides = std::index_sequence<1>;

        return tensor<const T, FlatShape, FlatStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data());
    } else {
        std::vector<size_t> flat_shape = {this->size()};
        std::vector<size_t> flat_strides = {1};

        return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), flat_shape, flat_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
void tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::reshape(std::vector<size_t> new_shape,
                                                                                   layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
        if (new_size != this->size()) {
            throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
        }
    }

    shape_ = std::move(new_shape);
    strides_ = compute_strides(l);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
void tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::reshape(std::vector<size_t> new_shape,
                                                                                   layout l) const
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
        if (new_size != this->size()) {
            throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
        }
    }

    // For const version, we can't modify the tensor itself, so we return a new view
    return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                  MemorySpace>(this->data(), new_shape, compute_strides(l));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::transpose() {
    if constexpr (fixed_shape<Shape>) {
        return tensor<T, reverse_sequence_t<Shape>, reverse_sequence_t<Strides>, ErrorChecking,
                      ownership_type::reference, MemorySpace>(this->data());
    } else {
        std::vector<size_t> reversed_shape = this->shape();
        std::reverse(reversed_shape.begin(), reversed_shape.end());
        std::vector<size_t> reversed_strides = this->strides();
        std::reverse(reversed_strides.begin(), reversed_strides.end());
        return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), reversed_shape, reversed_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::transpose() const {
    if constexpr (fixed_shape<Shape>) {
        return tensor<const T, reverse_sequence_t<Shape>, reverse_sequence_t<Strides>, ErrorChecking,
                      ownership_type::reference, MemorySpace>(this->data());
    } else {
        std::vector<size_t> reversed_shape = this->shape();
        std::reverse(reversed_shape.begin(), reversed_shape.end());
        std::vector<size_t> reversed_strides = this->strides();
        std::reverse(reversed_strides.begin(), reversed_strides.end());
        return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), reversed_shape, reversed_strides);
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_SHAPE_MANIPULATIONS_HPP