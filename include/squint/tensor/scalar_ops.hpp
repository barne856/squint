#ifndef SQUINT_TENSOR_SCALAR_OPS_HPP
#define SQUINT_TENSOR_SCALAR_OPS_HPP

#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <dimensionless_scalar U>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator*=(const U &s) -> tensor & {
    for (auto &element : *this) {
        element *= s;
    }
    return *this;
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <dimensionless_scalar U>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator/=(const U &s) -> tensor & {
    for (auto &element : *this) {
        element /= s;
    }
    return *this;
}

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

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace, scalar U>
auto operator*(const U &s, const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t)
    -> tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking, ownership_type::owner,
              MemorySpace> {
    return t * s;
}

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