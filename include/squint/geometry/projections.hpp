#ifndef SQUINT_GEOMETRY_PROJECTIONS_HPP
#define SQUINT_GEOMETRY_PROJECTIONS_HPP

#include "squint/quantity/quantity_types.hpp"
#include "squint/tensor/tensor.hpp"
#include <cmath>

namespace squint::geometry {

// Orthographic Projection
template <typename T>
auto ortho(length_t<T> left, length_t<T> right, length_t<T> bottom, length_t<T> top, length_t<T> near_plane,
           length_t<T> far_plane, length_t<T> unit_length = length_t<T>{1}) {
    tensor<T, shape<4, 4>> result;
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
template <dimensionless_scalar T, typename U>
auto perspective(T fovy, T aspect, length_t<U> near_plane, length_t<U> far_plane,
                 length_t<U> unit_length = length_t<U>{1}) {
    tensor<U, shape<4, 4>> result;
    U tanHalfFovy = std::tan(fovy / T{2});
    auto depth = far_plane - near_plane;

    result(0, 0) = U{1} / (aspect * tanHalfFovy);
    result(1, 1) = U{1} / tanHalfFovy;
    result(2, 2) = -(far_plane) / depth;
    result(2, 3) = (-far_plane * near_plane) / (depth * unit_length);
    result(3, 2) = -U{1};

    return result;
}

} // namespace squint::geometry

#endif // SQUINT_GEOMETRY_PROJECTIONS_HPP