/**
 * @file tensor_element_access.hpp
 * @brief Implementation of tensor class element access methods.
 *
 * This file contains the implementations of various element access methods for the tensor class,
 * including methods to access individual elements using indices, operator(), and operator[].
 * It provides both const and non-const access and handles multidimensional access.
 *
 */

#ifndef SQUINT_TENSOR_TENSOR_ELEMENT_ACCESS_HPP
#define SQUINT_TENSOR_TENSOR_ELEMENT_ACCESS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::access_element(
    const index_type &indices) const -> const T & {
    if constexpr (ErrorChecking == error_checking::enabled) {
        check_bounds(indices);
    }
    return data()[compute_offset(indices)];
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator()(Indices... indices) const
    -> const T & {
    return access_element({static_cast<std::size_t>(indices)...});
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator()(Indices... indices) -> T & {
    return const_cast<T &>(std::as_const(*this)(indices...));
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator[](const index_type &indices) const
    -> const T & {
    return access_element(indices);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator[](const index_type &indices)
    -> T & {
    return const_cast<T &>(std::as_const(*this)[indices]);
}

#ifndef _MSC_VER
// MSVC does not support the multidimensional subscript operator yet

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator[](Indices... indices) const
    -> const T & {
    return access_element({static_cast<std::size_t>(indices)...});
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <typename... Indices>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator[](Indices... indices) -> T & {
    return const_cast<T &>(std::as_const(*this)[indices...]);
}

#endif // !_MSC_VER

// Private helper methods

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <std::size_t... Is>
[[nodiscard]] constexpr auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::compute_offset_impl(
    const index_type &indices, std::index_sequence<Is...> /*unused*/) const -> std::size_t {
    return ((indices[Is] * std::get<Is>(make_array(Strides{}))) + ... + 0);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
[[nodiscard]] constexpr auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::compute_offset(
    const index_type &indices) const -> std::size_t {
    if constexpr (fixed_shape<Strides>) {
        return compute_offset_impl(indices, std::make_index_sequence<Strides::size()>{});
    } else {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < rank(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
constexpr auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::check_bounds(
    const index_type &indices) const -> void {
    if (indices.size() != rank()) {
        throw std::out_of_range("Invalid number of indices");
    }
    for (std::size_t i = 0; i < rank(); ++i) {
        if (indices[i] >= shape()[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_ELEMENT_ACCESS_HPP