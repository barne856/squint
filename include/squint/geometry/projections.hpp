#ifndef SQUINT_GEOMETRY_PROJECTIONS_HPP
#define SQUINT_GEOMETRY_PROJECTIONS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/layout.hpp"
#include "squint/quantity/quantity_types.hpp"
#include "squint/tensor/tensor.hpp"
#include <cmath>

namespace squint::geometry {

/**
 * @brief Creates an orthographic projection matrix.
 *
 * This function generates a 4x4 orthographic projection matrix that maps the specified
 * viewing frustum onto a unit cube centered at the origin.
 *
 * @tparam T The underlying scalar type for the length quantities.
 * @param left The left clipping plane coordinate.
 * @param right The right clipping plane coordinate.
 * @param bottom The bottom clipping plane coordinate.
 * @param top The top clipping plane coordinate.
 * @param near_plane The near clipping plane distance.
 * @param far_plane The far clipping plane distance.
 * @param unit_length The unit length for the projection space (default is 1).
 * @return A 4x4 tensor representing the orthographic projection matrix.
 *
 * @note The resulting matrix assumes a right-handed coordinate system.
 */
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

/**
 * @brief Creates a perspective projection matrix.
 *
 * This function generates a 4x4 perspective projection matrix based on the specified
 * field of view, aspect ratio, and near and far clipping planes.
 *
 * @tparam T The underlying scalar type for the field of view and aspect ratio.
 * @tparam U The underlying scalar type for the length quantities.
 * @param fovy The vertical field of view in radians.
 * @param aspect The aspect ratio (width / height) of the viewport.
 * @param near_plane The distance to the near clipping plane.
 * @param far_plane The distance to the far clipping plane.
 * @param unit_length The unit length for the projection space (default is 1).
 * @return A 4x4 tensor representing the perspective projection matrix.
 *
 * @note The resulting matrix assumes a right-handed coordinate system.
 * @note The field of view (fovy) should be in radians.
 */
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