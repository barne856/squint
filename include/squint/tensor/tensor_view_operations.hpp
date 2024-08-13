/**
 * @file tensor_view_operations.hpp
 * @brief Implementation of tensor class view and subview operations.
 *
 * This file contains the implementations of various view and subview operations for the tensor class,
 * including methods to create subviews with fixed and dynamic shapes, full tensor views,
 * and diagonal views. It provides both const and non-const versions of these operations.
 *
 */

#ifndef SQUINT_TENSOR_TENSOR_VIEW_OPERATIONS_HPP
#define SQUINT_TENSOR_TENSOR_VIEW_OPERATIONS_HPP

#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"
#include <stdexcept>

namespace squint {

// Subview operations

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename SubviewShape, typename StepSizes>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(const index_type &start_indices)
    requires fixed_shape<Shape>
{
    static_assert(SubviewShape::size() <= Shape::size(),
                  "Subview dimensions must be less than or equal to tensor rank");
    static_assert(StepSizes::size() <= Shape::size(), "Step sizes must match subview rank");
    using SubviewStrides =
        multiply_sequences_t<remove_last_n_t<Strides, Shape::size() - SubviewShape::size()>, StepSizes>;
    return tensor<T, SubviewShape, SubviewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
        data() + compute_offset(start_indices));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename SubviewShape, typename StepSizes>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(
    const index_type &start_indices) const
    requires fixed_shape<Shape>
{
    static_assert(SubviewShape::size() <= Shape::size(),
                  "Subview dimensions must be less than or equal to tensor rank");
    static_assert(StepSizes::size() <= Shape::size(), "Step sizes must match subview rank");
    using SubviewStrides =
        multiply_sequences_t<remove_last_n_t<Strides, Shape::size() - SubviewShape::size()>, StepSizes>;
    return tensor<const T, SubviewShape, SubviewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
        data() + compute_offset(start_indices));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <std::size_t... Dims, typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(Indices... start_indices)
    requires fixed_shape<Shape>
{
    static_assert(sizeof...(Indices) == Shape::size(), "Number of indices must match tensor rank");
    using StepSizes = repeat_sequence_t<sizeof...(Dims), 1>;
    return this->template subview<std::index_sequence<Dims...>, StepSizes>(
        {static_cast<std::size_t>(start_indices)...});
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <std::size_t... Dims, typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(Indices... start_indices) const
    requires fixed_shape<Shape>
{
    static_assert(sizeof...(Indices) == Shape::size(), "Number of indices must match tensor rank");
    using StepSizes = repeat_sequence_t<sizeof...(Dims), 1>;
    return this->template subview<std::index_sequence<Dims...>, StepSizes>(
        {static_cast<std::size_t>(start_indices)...});
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(const index_type &subview_shape,
                                                                                   const index_type &start_indices)
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (start_indices.size() != rank() || subview_shape.size() > rank()) {
            throw std::invalid_argument("Invalid subview dimensions");
        }
    }
    auto subview_strides = strides_;
    subview_strides.resize(subview_shape.size());
    return tensor<T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking, ownership_type::reference,
                  MemorySpace>(data() + compute_offset(start_indices), subview_shape, subview_strides);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(const index_type &subview_shape,
                                                                                   const index_type &start_indices) const
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (start_indices.size() != rank() || subview_shape.size() > rank()) {
            throw std::invalid_argument("Invalid subview dimensions");
        }
    }
    auto subview_strides = strides_;
    subview_strides.resize(subview_shape.size());
    return tensor<const T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking, ownership_type::reference,
                  MemorySpace>(data() + compute_offset(start_indices), subview_shape, subview_strides);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(const index_type &subview_shape,
                                                                                   const index_type &start_indices,
                                                                                   const index_type &step_sizes)
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (start_indices.size() != rank() || subview_shape.size() > rank()) {
            throw std::invalid_argument("Invalid subview dimensions");
        }
    }
    auto subview_strides = strides_;
    subview_strides.resize(subview_shape.size());
    for (size_t i = 0; i < subview_shape.size(); ++i) {
        subview_strides[i] *= step_sizes[i];
    }
    return tensor<T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking, ownership_type::reference,
                  MemorySpace>(data() + compute_offset(start_indices), subview_shape, subview_strides);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subview(const index_type &subview_shape,
                                                                                   const index_type &start_indices,
                                                                                   const index_type &step_sizes) const
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (start_indices.size() != rank() || subview_shape.size() > rank()) {
            throw std::invalid_argument("Invalid subview dimensions");
        }
    }
    auto subview_strides = strides_;
    subview_strides.resize(subview_shape.size());
    for (size_t i = 0; i < subview_shape.size(); ++i) {
        subview_strides[i] *= step_sizes[i];
    }
    return tensor<const T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking, ownership_type::reference,
                  MemorySpace>(data() + compute_offset(start_indices), subview_shape, subview_strides);
}

// View operations

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::view() {
    if constexpr (fixed_shape<Shape>) {
        return tensor<T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    } else {
        return tensor<T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data(), this->shape(), this->strides());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::view() const {
    if constexpr (fixed_shape<Shape>) {
        return tensor<const T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    } else {
        return tensor<const T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data(), this->shape(), this->strides());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::diag_view() {
    if constexpr (fixed_shape<Shape>) {
        static_assert(all_equal(Shape{}), "Diagonal view is only valid for square tensors");
        constexpr size_t diag_size = std::get<0>(make_array(Shape{}));
        using DiagShape = std::index_sequence<diag_size>;
        using DiagStrides = std::index_sequence<sum(Strides{})>;
        return tensor<T, DiagShape, DiagStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    } else {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (!std::all_of(this->shape().begin(), this->shape().end(),
                             [this](size_t s) { return s == this->shape()[0]; })) {
                throw std::invalid_argument("Diagonal view is only valid for square tensors");
            }
        }
        std::vector<size_t> diag_shape = {this->shape()[0]};
        std::vector<size_t> diag_strides = {std::accumulate(this->strides().begin(), this->strides().end(), 0ULL)};
        return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), diag_shape, diag_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::diag_view() const {
    if constexpr (fixed_shape<Shape>) {
        static_assert(all_equal(Shape{}), "Diagonal view is only valid for square tensors");
        constexpr size_t diag_size = std::get<0>(make_array(Shape{}));
        using DiagShape = std::index_sequence<diag_size>;
        using DiagStrides = std::index_sequence<sum(Strides{})>;
        return tensor<const T, DiagShape, DiagStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data());
    } else {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (!std::all_of(this->shape().begin(), this->shape().end(),
                             [this](size_t s) { return s == this->shape()[0]; })) {
                throw std::invalid_argument("Diagonal view is only valid for square tensors");
            }
        }
        std::vector<size_t> diag_shape = {this->shape()[0]};
        std::vector<size_t> diag_strides = {std::accumulate(this->strides().begin(), this->strides().end(), 0ULL)};
        return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data(), diag_shape, diag_strides);
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_VIEW_OPERATIONS_HPP