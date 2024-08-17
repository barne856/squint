/**
 * @file scalar_ops.hpp
 * @brief Scalar operations for tensor objects.
 *
 * This file contains implementations of scalar operations on tensors,
 * including multiplication and division by scalars.
 */
#ifndef SQUINT_TENSOR_SCALAR_OPS_HPP
#define SQUINT_TENSOR_SCALAR_OPS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"

namespace squint {

// Scalar multiplication assignment
/**
 * @brief Scalar multiplication assignment operator.
 * @param s The scalar to multiply the tensor by.
 * @return Reference to the modified tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <dimensionless_scalar U>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator*=(const U &s) -> tensor & {
    for (auto &element : *this) {
        element *= s;
    }
    return *this;
}

// Scalar division assignment
/**
 * @brief Scalar division assignment operator.
 * @param s The scalar to divide the tensor by.
 * @return Reference to the modified tensor.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <dimensionless_scalar U>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator/=(const U &s) -> tensor & {
    for (auto &element : *this) {
        element /= s;
    }
    return *this;
}

// Tensor-scalar multiplication
/**
 * @brief Tensor-scalar multiplication operator.
 * @param t The tensor to be multiplied.
 * @param s The scalar to multiply by.
 * @return A new tensor containing the result of the multiplication.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, scalar U>
auto operator*(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t,
               const U &s) -> tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                                     ownership_type::owner, MemorySpace> {
    using result_type = tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                               ownership_type::owner, MemorySpace>;
    if constexpr (fixed_shape<Shape>) {
        result_type result;
        auto result_it = result.begin();
        for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
            *result_it = *it * s;
        }
        return result;
    } else {
        result_type result(t.shape());
        auto result_it = result.begin();
        for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
            *result_it = *it * s;
        }
        return result;
    }
}

// Scalar-tensor multiplication
/**
 * @brief Scalar-tensor multiplication operator.
 * @param s The scalar to multiply by.
 * @param t The tensor to be multiplied.
 * @return A new tensor containing the result of the multiplication.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, scalar U>
auto operator*(const U &s, const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t)
    -> tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking, ownership_type::owner,
              MemorySpace> {
    return t * s;
}

// Tensor-scalar division
/**
 * @brief Tensor-scalar division operator.
 * @param t The tensor to be divided.
 * @param s The scalar to divide by.
 * @return A new tensor containing the result of the division.
 */
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, scalar U>
auto operator/(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t,
               const U &s) -> tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                                     ownership_type::owner, MemorySpace> {
    using result_type = tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                               ownership_type::owner, MemorySpace>;
    if constexpr (fixed_shape<Shape>) {
        result_type result;
        auto result_it = result.begin();
        for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
            *result_it = *it / s;
        }
        return result;
    } else {
        result_type result(t.shape());
        auto result_it = result.begin();
        for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
            *result_it = *it / s;
        }
        return result;
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_SCALAR_OPS_HPP