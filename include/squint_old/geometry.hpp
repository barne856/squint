#ifndef SQUINT_GEOMETRY_HPP
#define SQUINT_GEOMETRY_HPP

#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <concepts>

namespace squint {

template <typename T>
concept dimensionless_scalar = scalar<T> && (arithmetic<T> || dimensionless_quantity<T>);

template <typename T>
concept transformation_matrix = fixed_shape_tensor<T> && dimensionless_tensor<T> && T::constexpr_shape().size() == 2 &&
                                T::constexpr_shape()[0] == 4 && T::constexpr_shape()[1] == 4;

// Affine Transformations
// Translation
template <transformation_matrix T, arithmetic U>
auto translate(T &matrix, const vec3_t<units::length_t<U>> &x, units::length_t<U> unit_length = units::length_t<U>{1}) {
    matrix.template subview<3, 1>(0, 3) += x / unit_length;
}

// Rotation (around arbitrary axis)
template <transformation_matrix T, dimensionless_scalar U> auto rotate(T &matrix, U angle, const vec3_t<U> &axis) {
    U c = std::cos(angle);
    U s = std::sin(angle);
    auto norm_axis = normalize(axis);
    U t = U{1} - c;
    auto rotation = mat4_t<U>::eye();
    auto A = mat3_t<U>::eye() * c;
    auto B = mat3_t<U>{
        {U{0}, norm_axis(2), -norm_axis(1), -norm_axis(2), U{0}, norm_axis(0), norm_axis(1), -norm_axis(0), U{0}}};
    auto C = t * (norm_axis * norm_axis.transpose());
    auto R = A + s * B + C;
    rotation.template subview<3, 3>(0,0) = R;
    matrix = rotation * matrix;
}

// Scale
template <transformation_matrix T, dimensionless_scalar U> auto scale(T &matrix, const vec3_t<U> &s) {
    matrix(0, 0) *= s(0);
    matrix(1, 1) *= s(1);
    matrix(2, 2) *= s(2);
}

// Orthographic Projection
template <arithmetic T>
auto ortho(units::length_t<T> left, units::length_t<T> right, units::length_t<T> bottom, units::length_t<T> top,
           units::length_t<T> near_plane, units::length_t<T> far_plane,
           units::length_t<T> unit_length = units::length_t<T>{1}) {
    auto result = mat4_t<T>{};
    auto width = (right - left);
    auto height = (top - bottom);
    auto depth = (far_plane - near_plane);

    result(0, 0) = unit_length * T{2} / width;
    result(1, 1) = unit_length * T{2} / height;
    result(2, 2) = unit_length * T{1} / depth;
    result(0, 3) = -(right + left) / width;
    result(1, 3) = -(top + bottom) / height;
    result(2, 3) = -near_plane / depth;
    result(3, 3) = T{1};

    return result;
}

// Perspective Projection
template <dimensionless_scalar T, arithmetic U>
auto perspective(T fovy, T aspect, units::length_t<U> near_plane, units::length_t<U> far_plane,
                 units::length_t<U> unit_length = units::length_t<U>{1}) {
    auto result = mat4_t<U>{};
    U tanHalfFovy = std::tan(fovy / T{2});
    auto depth = far_plane - near_plane;

    result(0, 0) = U{1} / (aspect * tanHalfFovy);
    result(1, 1) = U{1} / tanHalfFovy;
    result(2, 2) = -(far_plane) / depth;
    result(2, 3) = (-far_plane * near_plane) / (depth * unit_length);
    result(3, 2) = -U{1};

    return result;
}
} // namespace squint

#endif // SQUINT_GEOMETRY_HPP