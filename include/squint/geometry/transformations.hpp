#ifndef SQUINT_GEOMETRY_TRANSFORMATIONS_HPP
#define SQUINT_GEOMETRY_TRANSFORMATIONS_HPP

#include "squint/quantity/quantity_types.hpp"
#include "squint/tensor/tensor.hpp"

namespace squint::geometry {

template <typename T>
concept transformation_matrix = fixed_tensor<T> && dimensionless_scalar<typename T::value_type> &&
                                std::is_same_v<typename T::shape_type, shape<4, 4>>;

// Translation
template <transformation_matrix T, typename U>
void translate(T &matrix, const tensor<length_t<U>, shape<3>> &x, length_t<U> unit_length = length_t<U>{1}) {
    matrix.template subview<3, 1>(0, 3) += x / unit_length;
}

// Rotation (around arbitrary axis)
template <transformation_matrix T, dimensionless_scalar U>
void rotate(T &matrix, U angle, const tensor<U, shape<3>> &axis) {
    U c = std::cos(angle);
    U s = std::sin(angle);
    auto norm_axis = axis / norm(axis);
    U t = U{1} - c;
    auto rotation = tensor<U, shape<4, 4>>::eye();
    auto A = tensor<U, shape<3, 3>>::eye() * c;
    auto B = tensor<U, shape<3, 3>>{
        {U{0}, norm_axis(2), -norm_axis(1), -norm_axis(2), U{0}, norm_axis(0), norm_axis(1), -norm_axis(0), U{0}}};
    auto C = t * (norm_axis * norm_axis.transpose());
    auto R = A + s * B + C;
    rotation.template subview<3, 3>(0, 0) = R;
    matrix = rotation * matrix;
}

// Scale
template <transformation_matrix T, dimensionless_scalar U> void scale(T &matrix, const tensor<U, shape<3>> &s) {
    matrix(0, 0) *= s(0);
    matrix(1, 1) *= s(1);
    matrix(2, 2) *= s(2);
}

} // namespace squint::geometry

#endif // SQUINT_GEOMETRY_TRANSFORMATIONS_HPP