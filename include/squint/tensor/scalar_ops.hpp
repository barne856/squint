#ifndef SQUINT_TENSOR_SCALAR_OPS_HPP
#define SQUINT_TENSOR_SCALAR_OPS_HPP

#include "squint/core/error_checking.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"

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
    tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking, ownership_type::owner,
           MemorySpace>
        result(t);
    result *= s;
    return result;
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
    tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking, ownership_type::owner,
           MemorySpace>
        result(t);
    result /= s;
    return result;
}

} // namespace squint

#endif // SQUINT_TENSOR_SCALAR_OPS_HPP