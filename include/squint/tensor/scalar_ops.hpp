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
// NOLINTNEXTLINE
#include "squint/tensor/tensor_op_compatibility.hpp"

#ifdef SQUINT_USE_CUDA
#include "squint/tensor/cuda/scalar.hpp"
#endif

namespace squint {

template <scalar T> auto get_scalar_value(const T &s) {
    if constexpr (quantitative<T>) {
        return s.value();
    } else {
        return s;
    }
}

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
    if constexpr (MemorySpace == memory_space::host) {
        for (auto &element : *this) {
            element *= s;
        }
    } else {
#ifdef SQUINT_USE_CUDA
        // NOLINTBEGIN
        using blas_type = blas_type_t<T>;
        blas_type scalar = static_cast<blas_type>(get_scalar_value(s));
        if constexpr (std::is_same_v<blas_type, float>) {
            scalar_multiplication<float>(scalar, reinterpret_cast<float *>(data()),
                                         reinterpret_cast<const float *>(data()), device_shape(), device_strides(),
                                         device_strides(), shape().size(), size());
        } else if constexpr (std::is_same_v<blas_type, double>) {
            scalar_multiplication<double>(scalar, reinterpret_cast<double *>(data()),
                                          reinterpret_cast<const double *>(data()), device_shape(), device_strides(),
                                          device_strides(), shape().size(), size());
        }
        // NOLINTEND
#endif
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
    if constexpr (MemorySpace == memory_space::host) {
        for (auto &element : *this) {
            element /= s;
        }
    } else {
#ifdef SQUINT_USE_CUDA
        // NOLINTBEGIN
        using blas_type = blas_type_t<T>;
        blas_type scalar = blas_type(1) / static_cast<blas_type>(get_scalar_value(s));
        if constexpr (std::is_same_v<blas_type, float>) {
            scalar_multiplication<float>(scalar, reinterpret_cast<float *>(data()),
                                         reinterpret_cast<const float *>(data()), device_shape(), device_strides(),
                                         device_strides(), shape().size(), size());
        } else if constexpr (std::is_same_v<blas_type, double>) {
            scalar_multiplication<double>(scalar, reinterpret_cast<double *>(data()),
                                          reinterpret_cast<const double *>(data()), device_shape(), device_strides(),
                                          device_strides(), shape().size(), size());
        }
        // NOLINTEND
#endif
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
auto operator*(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t, const U &s) {
    if constexpr (fixed_shape<Shape>) {
        if constexpr (MemorySpace == memory_space::host) {
            using result_type = tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::owner, MemorySpace>;
            result_type result{};
            auto result_it = result.begin();
            for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
                *result_it = *it * s;
            }
            return std::move(result);
        } else {
#ifdef SQUINT_USE_CUDA
            using result_type = tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::reference, MemorySpace>;
            result_type result{};
            // NOLINTBEGIN
            using blas_type = std::common_type_t<blas_type_t<T>, blas_type_t<U>>;
            blas_type scalar = static_cast<blas_type>(get_scalar_value(s));
            if constexpr (std::is_same_v<blas_type, float>) {
                scalar_multiplication<float>(scalar, reinterpret_cast<float *>(result.data()),
                                             reinterpret_cast<const float *>(t.data()), t.device_shape(),
                                             t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            } else if constexpr (std::is_same_v<blas_type, double>) {
                scalar_multiplication<double>(scalar, reinterpret_cast<double *>(result.data()),
                                              reinterpret_cast<const double *>(t.data()), t.device_shape(),
                                              t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            }
            // NOLINTEND
            return std::move(result);
#endif
        }
    } else {
        if constexpr (MemorySpace == memory_space::host) {
            using result_type = tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::owner, MemorySpace>;
            result_type result(t.shape());
            auto result_it = result.begin();
            for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
                *result_it = *it * s;
            }
            return std::move(result);
        } else {
#ifdef SQUINT_USE_CUDA
            using result_type = tensor<decltype(std::declval<T>() * std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::reference, MemorySpace>;
            result_type result(t.shape());
            // NOLINTBEGIN
            using blas_type = std::common_type_t<blas_type_t<T>, blas_type_t<U>>;
            blas_type scalar = static_cast<blas_type>(get_scalar_value(s));
            if constexpr (std::is_same_v<blas_type, float>) {
                scalar_multiplication<float>(scalar, reinterpret_cast<float *>(result.data()),
                                             reinterpret_cast<const float *>(t.data()), t.device_shape(),
                                             t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            } else if constexpr (std::is_same_v<blas_type, double>) {
                scalar_multiplication<double>(scalar, reinterpret_cast<double *>(result.data()),
                                              reinterpret_cast<const double *>(t.data()), t.device_shape(),
                                              t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            }
            // NOLINTEND
            return std::move(result);
#endif
        }
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
auto operator*(const U &s, const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t) {
    auto result = t * s;
    return std::move(result);
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
auto operator/(const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t, const U &s) {

    if constexpr (fixed_shape<Shape>) {
        if constexpr (MemorySpace == memory_space::host) {
            using result_type = tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::owner, MemorySpace>;
            result_type result{};
            auto result_it = result.begin();
            for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
                *result_it = *it / s;
            }
            return std::move(result);
        } else {
#ifdef SQUINT_USE_CUDA
            using result_type = tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::reference, MemorySpace>;
            result_type result{};
            // NOLINTBEGIN
            using blas_type = std::common_type_t<blas_type_t<T>, blas_type_t<U>>;
            blas_type scalar = blas_type(1) / static_cast<blas_type>(get_scalar_value(s));
            if constexpr (std::is_same_v<blas_type, float>) {
                scalar_multiplication<float>(scalar, reinterpret_cast<float *>(result.data()),
                                             reinterpret_cast<const float *>(t.data()), t.device_shape(),
                                             t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            } else if constexpr (std::is_same_v<blas_type, double>) {
                scalar_multiplication<double>(scalar, reinterpret_cast<double *>(result.data()),
                                              reinterpret_cast<const double *>(t.data()), t.device_shape(),
                                              t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            }
            // NOLINTEND
            return std::move(result);
#endif
        }
    } else {
        if constexpr (MemorySpace == memory_space::host) {
            using result_type = tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::owner, MemorySpace>;
            result_type result(t.shape());
            auto result_it = result.begin();
            for (auto it = t.begin(); it != t.end(); ++it, ++result_it) {
                *result_it = *it / s;
            }
            return std::move(result);
        } else {
#ifdef SQUINT_USE_CUDA
            using result_type = tensor<decltype(std::declval<T>() / std::declval<U>()), Shape, Strides, ErrorChecking,
                                       ownership_type::reference, MemorySpace>;
            result_type result(t.shape());
            // NOLINTBEGIN
            using blas_type = std::common_type_t<blas_type_t<T>, blas_type_t<U>>;
            blas_type scalar = blas_type(1) / static_cast<blas_type>(get_scalar_value(s));
            if constexpr (std::is_same_v<blas_type, float>) {
                scalar_multiplication<float>(scalar, reinterpret_cast<float *>(result.data()),
                                             reinterpret_cast<const float *>(t.data()), t.device_shape(),
                                             t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            } else if constexpr (std::is_same_v<blas_type, double>) {
                scalar_multiplication<double>(scalar, reinterpret_cast<double *>(result.data()),
                                              reinterpret_cast<const double *>(t.data()), t.device_shape(),
                                              t.device_strides(), t.device_strides(), t.shape().size(), t.size());
            }
            // NOLINTEND
            return std::move(result);
#endif
        }
    }
}

} // namespace squint

#endif // SQUINT_TENSOR_SCALAR_OPS_HPP