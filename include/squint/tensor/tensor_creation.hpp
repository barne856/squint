/**
 * @file tensor_creation.hpp
 * @brief Implementation of static tensor creation methods.
 *
 * This file contains the implementations of various static methods for creating tensors
 * with specific initial values or patterns. These methods include creating tensors filled
 * with zeros, ones, a specific value, random values, identity tensors, and diagonal tensors.
 *
 */

#ifndef SQUINT_TENSOR_TENSOR_CREATION_HPP
#define SQUINT_TENSOR_TENSOR_CREATION_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::zeros(const std::vector<size_t> &shape,
                                                                                 layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        return tensor(); // Default constructor initializes to zero for arithmetic types
    } else {
        return tensor(shape, l); // Uses the constructor we defined earlier
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::ones(const std::vector<size_t> &shape,
                                                                                layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        tensor t;
        std::fill(t.data(), t.data() + t.size(), T(1));
        return t;
    } else {
        tensor t(shape, l);
        std::fill(t.data(), t.data() + t.size(), T(1));
        return t;
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::full(const T &value,
                                                                                const std::vector<size_t> &shape,
                                                                                layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        tensor t;
        std::fill(t.data(), t.data() + t.size(), value);
        return t;
    } else {
        tensor t(shape, l);
        std::fill(t.data(), t.data() + t.size(), value);
        return t;
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::random(T min, T max,
                                                                                  const std::vector<size_t> &shape,
                                                                                  layout l)
    requires(OwnershipType == ownership_type::owner)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    if constexpr (fixed_shape<Shape>) {
        tensor t;
        std::generate(t.data(), t.data() + t.size(), [&]() { return dis(gen); });
        return t;
    } else {
        tensor t(shape, l);
        std::generate(t.data(), t.data() + t.size(), [&]() { return dis(gen); });
        return t;
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::eye(const std::vector<size_t> &shape,
                                                                               layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        constexpr auto dims = make_array(Shape{});
        static_assert(dims.size() == 2 && dims[0] == dims[1], "Eye tensor must be square");
        tensor t{};
        for (size_t i = 0; i < dims[0]; ++i) {
            t(i, i) = T(1);
        }
        return t;
    } else {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (shape.size() != 2 || shape[0] != shape[1]) {
                throw std::invalid_argument("Eye tensor must be square");
            }
        }
        tensor t(shape, l);
        for (size_t i = 0; i < shape[0]; ++i) {
            t(i, i) = T(1);
        }
        return t;
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::diag(const T &value,
                                                                                const std::vector<size_t> &shape,
                                                                                layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        constexpr auto dims = make_array(Shape{});
        static_assert(dims.size() == 2 && dims[0] == dims[1], "Diagonal tensor must be square");
        tensor t{};
        for (size_t i = 0; i < dims[0]; ++i) {
            t(i, i) = value;
        }
        return t;
    } else {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (shape.size() != 2 || shape[0] != shape[1]) {
                throw std::invalid_argument("Diagonal tensor must be square");
            }
        }
        tensor t(shape, l);
        for (size_t i = 0; i < shape[0]; ++i) {
            t(i, i) = value;
        }
        return t;
    }
}

// arange
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::arange(T start, T step,
                                                                                  const std::vector<size_t> &shape,
                                                                                  layout l)
    requires(OwnershipType == ownership_type::owner)
{
    if constexpr (fixed_shape<Shape>) {
        tensor t;
        // fill the tensor with the values
        T value = start;
        for (auto &elem : t) {
            elem = value;
            value += step;
        }
        return t;
    } else {
        tensor t(shape, l);
        // fill the tensor with the values
        T value = start;
        for (auto &elem : t) {
            elem = value;
            value += step;
        }
        return t;
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_CREATION_HPP