/**
 * @file tensor_types.hpp
 * @brief Type aliases for creating tensors with specific types and dimensions.
 *
 * This file contains type aliases for creating tensors with specific types and dimensions.
 * These aliases simplify the creation of tensor objects with specific data types and shapes.
 */
#ifndef SQUINT_TENSOR_TENSOR_TYPES_HPP
#define SQUINT_TENSOR_TENSOR_TYPES_HPP

#include "squint/core/layout.hpp"
#include "squint/tensor/tensor.hpp"

#include <cstddef>

namespace squint {

// Vector types
template <typename T> using vec2_t = tensor<T, shape<2>>;
template <typename T> using vec3_t = tensor<T, shape<3>>;
template <typename T> using vec4_t = tensor<T, shape<4>>;
using ivec2 = vec2_t<int>;
using ivec3 = vec3_t<int>;
using ivec4 = vec4_t<int>;
using uvec2 = vec2_t<unsigned char>;
using uvec3 = vec3_t<unsigned char>;
using uvec4 = vec4_t<unsigned char>;
using vec2 = vec2_t<float>;
using vec3 = vec3_t<float>;
using vec4 = vec4_t<float>;
using dvec2 = vec2_t<double>;
using dvec3 = vec3_t<double>;
using dvec4 = vec4_t<double>;
using bvec2 = vec2_t<bool>;
using bvec3 = vec3_t<bool>;
using bvec4 = vec4_t<bool>;

// Square matrix types
template <typename T> using mat2_t = tensor<T, shape<2, 2>>;
template <typename T> using mat3_t = tensor<T, shape<3, 3>>;
template <typename T> using mat4_t = tensor<T, shape<4, 4>>;
using imat2 = mat2_t<int>;
using imat3 = mat3_t<int>;
using imat4 = mat4_t<int>;
using umat2 = mat2_t<unsigned char>;
using umat3 = mat3_t<unsigned char>;
using umat4 = mat4_t<unsigned char>;
using mat2 = mat2_t<float>;
using mat3 = mat3_t<float>;
using mat4 = mat4_t<float>;
using dmat2 = mat2_t<double>;
using dmat3 = mat3_t<double>;
using dmat4 = mat4_t<double>;
using bmat2 = mat2_t<bool>;
using bmat3 = mat3_t<bool>;
using bmat4 = mat4_t<bool>;

// Non-square matrix types
template <typename T> using mat2x3_t = tensor<T, shape<2, 3>>;
template <typename T> using mat2x4_t = tensor<T, shape<2, 4>>;
template <typename T> using mat3x2_t = tensor<T, shape<3, 2>>;
template <typename T> using mat3x4_t = tensor<T, shape<3, 4>>;
template <typename T> using mat4x2_t = tensor<T, shape<4, 2>>;
template <typename T> using mat4x3_t = tensor<T, shape<4, 3>>;
using imat2x3 = mat2x3_t<int>;
using imat2x4 = mat2x4_t<int>;
using imat3x2 = mat3x2_t<int>;
using imat3x4 = mat3x4_t<int>;
using imat4x2 = mat4x2_t<int>;
using imat4x3 = mat4x3_t<int>;
using umat2x3 = mat2x3_t<unsigned char>;
using umat2x4 = mat2x4_t<unsigned char>;
using umat3x2 = mat3x2_t<unsigned char>;
using umat3x4 = mat3x4_t<unsigned char>;
using umat4x2 = mat4x2_t<unsigned char>;
using umat4x3 = mat4x3_t<unsigned char>;
using mat2x3 = mat2x3_t<float>;
using mat2x4 = mat2x4_t<float>;
using mat3x2 = mat3x2_t<float>;
using mat3x4 = mat3x4_t<float>;
using mat4x2 = mat4x2_t<float>;
using mat4x3 = mat4x3_t<float>;
using dmat2x3 = mat2x3_t<double>;
using dmat2x4 = mat2x4_t<double>;
using dmat3x2 = mat3x2_t<double>;
using dmat3x4 = mat3x4_t<double>;
using dmat4x2 = mat4x2_t<double>;
using dmat4x3 = mat4x3_t<double>;
using bmat2x3 = mat2x3_t<bool>;
using bmat2x4 = mat2x4_t<bool>;
using bmat3x2 = mat3x2_t<bool>;
using bmat3x4 = mat3x4_t<bool>;
using bmat4x2 = mat4x2_t<bool>;
using bmat4x3 = mat4x3_t<bool>;

// General tensor shapes
template <typename T, std::size_t... Dims> using ndarr_t = tensor<T, shape<Dims...>>;
template <std::size_t... Dims> using indarr = ndarr_t<int, Dims...>;
template <std::size_t... Dims> using undarr = ndarr_t<unsigned char, Dims...>;
template <std::size_t... Dims> using ndarr = ndarr_t<float, Dims...>;
template <std::size_t... Dims> using dndarr = ndarr_t<double, Dims...>;
template <std::size_t... Dims> using bndarr = ndarr_t<bool, Dims...>;

template <typename T> using tens_t = tensor<T, dynamic, dynamic>;
using itens = tens_t<int>;
using utens = tens_t<unsigned char>;
using tens = tens_t<float>;
using dtens = tens_t<double>;
using btens = tens_t<bool>;

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_TYPES_HPP