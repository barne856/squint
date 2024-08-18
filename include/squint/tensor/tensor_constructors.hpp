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

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_iteration.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

// Initializer list constructor for fixed shape tensors
/**
 * @brief Constructs a tensor from an initializer list.
 * @param init The initializer list containing the tensor elements.
 * @throws std::invalid_argument if the initializer list size doesn't match the tensor size (when error checking is
 * enabled).
 */
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

// Single value constructor for fixed shape tensors
/**
 * @brief Constructs a tensor filled with a single value.
 * @param value The value to fill the tensor with.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(const T &value)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
    : data_() {
    std::fill(data_.begin(), data_.end(), value);
}

// Array constructor for fixed shape tensors
/**
 * @brief Constructs a tensor from a std::array.
 * @param elements The array containing the tensor elements.
 * @throws std::invalid_argument if the array size doesn't match the tensor size (when error checking is enabled).
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(const std::array<T, _size()> &elements)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
    : data_(elements) {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (elements.size() != product(Shape{})) {
            throw std::invalid_argument("Input array size does not match tensor size");
        }
    }
}

// Constructor from other fixed tensors
/**
 * @brief Constructs a tensor from other fixed tensors.
 * @param ts The tensors to construct from.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <fixed_tensor... OtherTensor>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(const OtherTensor &...ts)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
{
    using OtherShape = typename std::common_type_t<OtherTensor...>::shape_type;
    static_assert(subview_compatible<tensor, OtherShape>(), "Incompatible tensor shapes");
    auto blocks = subviews<OtherShape>().begin();
    ((*blocks++ = ts), ...);
}

// Constructor for dynamic shape tensors with explicit shape and strides
/**
 * @brief Constructs a dynamic shape tensor with given shape and strides.
 * @param shape The shape of the tensor.
 * @param strides The strides of the tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(Shape shape, Strides strides)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(std::move(strides)) {
    data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
}

// Constructor for dynamic shape tensors with shape and layout
/**
 * @brief Constructs a dynamic shape tensor with given shape and layout.
 * @param shape The shape of the tensor.
 * @param l The layout of the tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(Shape shape, layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(compute_strides(l)) {
    data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
}

// Constructor for dynamic shape tensors with shape, elements, and layout
/**
 * @brief Constructs a dynamic shape tensor with given shape, elements, and layout.
 * @param shape The shape of the tensor.
 * @param elements The elements of the tensor.
 * @param l The layout of the tensor.
 * @throws std::invalid_argument if the elements size doesn't match the tensor size (when error checking is enabled).
 */
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

// Constructor for dynamic shape tensors with shape, single value, and layout
/**
 * @brief Constructs a dynamic shape tensor filled with a single value.
 * @param shape The shape of the tensor.
 * @param value The value to fill the tensor with.
 * @param l The layout of the tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(std::vector<size_t> shape, const T &value,
                                                                             layout l)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
    : shape_(std::move(shape)), strides_(compute_strides(l)) {
    size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
    data_.resize(total_size, value);
}

/**
 * @brief Constructs a tensor from another tensor of a different shape.
 * @param other The tensor to construct from.
 *
 * This constructor allows for implicit conversion to a tensor of the same shape with trailing 1's removed.
 */
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
        auto other_begin = other.begin();
        for (auto &element : *this) {
            element = *other_begin++;
        }
    } else {
        // for reference ownership, both strides and shape must be convertible
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
        static_assert(implicit_convertible_strides_v<Strides, OtherStrides>, "Invalid strides conversion");
        data_ = other.data();
    }
}

/**
 * @brief Constructs a tensor from another tensor of a different shape.
 * @param other The tensor to construct from.
 *
 * This constructor allows implicit conversion for compatible tensors of various shape from reference ownership type to
 * owner ownership type
 */
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
        if (ErrorChecking == error_checking::enabled) {
            if (!implicit_convertible_shapes_vector(other.shape(), shape())) {
                throw std::runtime_error("Invalid shape conversion");
            }
        }
        shape_ = other.shape();
        strides_ = other.strides();
        data_.resize(other.size());
        auto other_begin = other.begin();
        for (auto &element : *this) {
            element = *other_begin++;
        }
    } else {
        static_assert(implicit_convertible_shapes_v<Shape, OtherShape>, "Invalid shape conversion");
        auto other_begin = other.begin();
        for (auto &element : *this) {
            element = *other_begin++;
        }
    }
}

/**
 * @brief Constructs a tensor from another tensor of a different shape.
 * @param data The pointer to the data.
 * @param shape The shape of the tensor.
 * @param strides The strides of the tensor.
 *
 * This constructor constructs a tensor with reference ownership type from a pointer to data and a shape and strides.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(T *data, Shape shape, Strides strides)
    requires(dynamic_shape<Shape> && OwnershipType == ownership_type::reference)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
    if (ErrorChecking == error_checking::enabled) {
        if (!implicit_convertible_shapes_vector(shape, this->shape())) {
            throw std::runtime_error("Invalid shape conversion");
        }
    }
}

/**
 * @brief Constructs a tensor from a pointer to data.
 * @param data The pointer to the data.
 *
 * This constructor constructs a tensor with reference ownership type from a pointer to data.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::tensor(T *data)
    requires(fixed_shape<Shape> && OwnershipType == ownership_type::reference)
    : data_(data) {}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_CONSTRUCTORS_HPP