/**
 * @file linalg.hpp
 * @author Brendan Barnes
 * @brief Mathematical operator overloads and functions for use with tensors
 *
 * @copyright Copyright (c) 2022
 *
 */
module;
#ifdef SQUINT_USE_MKL
#include "mkl_service.h"
#include <mkl.h>
#include <mkl_blas.h>    // Intel MKL BLAS methods
#include <mkl_lapacke.h> // Inlet MKL LAPACKE methods
#endif
#include <ostream>
#include <cassert>
export module squint:linalg;
import :quantity;
import :tensor;
import :dimension;

export namespace squint {
// constexpr max
template <class T> inline constexpr T const &maximum(const T &first, const T &second) {
    return first < second ? second : first;
}

// used to check if two floats are approximatly equal
template <typename T>
    requires std::floating_point<T>
bool approx_equal(T a, T b, T epsilon = 128 * 1.192092896e-04, T abs_th = std::numeric_limits<T>::epsilon()) {
    assert(std::numeric_limits<T>::epsilon() <= epsilon);
    assert(epsilon < 1.F);

    if (a == b)
        return true;

    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
    return diff < std::max(abs_th, epsilon * norm);
}
// is close for floating point tensors
template <tensorial U, tensorial V>
    requires(!scalar<U> && !scalar<V>)
bool approx_equal(U const &x, V const &y) {
    assert(x.shape() == y.shape());
    auto it2 = y.begin();
    for (auto it1 = x.begin(); it2 != y.end(); it1++, it2++) {
        if (!approx_equal(*it1, *it2)) {
            return false;
        }
    }
    return true;
}
template <scalar U, scalar V>
    requires(tensor_shape<U> && tensor_shape<V>)
bool approx_equal(U const &x, V const &y) {
    return static_cast<bool>(approx_equal(x.data()[0], y.data()[0]));
}
template <quantitative U, quantitative V> bool approx_equal(U const &x, V const &y) {
    return static_cast<bool>(approx_equal(x._elem, y._elem));
}
template <scalar U> bool approx_equal(U const &x, typename U::value_type const &y) {
    return static_cast<bool>(approx_equal(x.data()[0], y));
}
template <scalar U> bool approx_equal(typename U::value_type const &x, U const &y) {
    return static_cast<bool>(approx_equal(x, y.data()[0]));
}
// nth-root
template <std::integral auto const N, typename T> T root(const T &val) { return std::pow(val, 1. / N); }
template <std::integral auto const N, scalar S>
    requires(tensorial<S>)
S root(const S &val) {
    return S(std::pow(val.data()[0], 1. / N));
}
template <std::integral auto const N, quantitative Q> auto root(const Q &val) {
    return quantities::quantity<typename Q::value_type, root_t<typename Q::dimension_type, N>>(
        std::pow(val._elem, 1. / N));
}
template <typename T, std::integral auto const N>
using root_type = decltype(root<N>(std::declval<typename T::value_type &>()));
// pow
template <std::integral auto const N, typename T> T pow(const T &val) { return std::pow(val, N); }
template <std::integral auto const N, scalar S>
    requires(tensorial<S>)
S pow(const S &val) {
    return S(std::sqrt(val.data()[0], N));
}
template <std::integral auto const N, quantitative Q> auto pow(const Q &val) {
    return quantities::quantity<typename Q::value_type, pow_t<typename Q::dimension_type, N>>(std::pow(val._elem, N));
}
template <typename T, std::integral auto const N>
using pow_type = decltype(pow<N>(std::declval<typename T::value_type &>()));
// +, -
template <fixed_tensor U, fixed_tensor V>
    requires(!scalar<U> && !scalar<V>)
auto operator+(const U &a, const V &b) {
    static_assert(same_shape<U, V>(), "tensors must have the same shape");
    using res_type = typename std::remove_const<sum_type<U, V>>::type;
    constexpr size_t N = U::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<res_type, U::shape()[I]...> c;
        auto it2 = b.begin();
        auto it3 = c.begin();
        for (auto it1 = a.begin(); it2 != b.end(); it1++, it2++, it3++) {
            *it3 = *it1 + *it2;
        }
        return c;
    }(std::make_index_sequence<N>{});
}
template <fixed_tensor U, fixed_tensor V>
    requires(!scalar<U> && !scalar<V>)
auto operator-(const U &a, const V &b) {
    static_assert(same_shape<U, V>(), "tensors must have the same shape");
    using res_type = typename std::remove_const<diff_type<U, V>>::type;
    constexpr size_t N = U::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<res_type, U::shape()[I]...> c;
        auto it2 = b.begin();
        auto it3 = c.begin();
        for (auto it1 = a.begin(); it2 != b.end(); it1++, it2++, it3++) {
            *it3 = *it1 - *it2;
        }
        return c;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor U, dynamic_tensor V> auto operator+(const U &a, const V &b) {
    assert(same_shape(a, b));
    using res_type = typename std::remove_const<sum_type<U, V>>::type;
    tensor<res_type, dynamic_shape> y(a.shape(), res_type(0));
    auto it3 = y.begin();
    auto it2 = b.begin();
    for (auto it1 = a.begin(); it2 != b.end(); it1++, it2++, it3++) {
        *it3 = *it1 + *it2;
    }
    return y;
}
template <dynamic_tensor U, dynamic_tensor V> auto operator-(const U &a, const V &b) {
    assert(same_shape(a, b));
    using res_type = typename std::remove_const<diff_type<U, V>>::type;
    tensor<res_type, dynamic_shape> y(a.shape(), res_type(0));
    auto it3 = y.begin();
    auto it2 = b.begin();
    for (auto it1 = a.begin(); it2 != b.end(); it1++, it2++, it3++) {
        *it3 = *it1 - *it2;
    }
    return y;
}
// norm
template <tensorial T> typename T::value_type vector_norm(const T &tens) {
    pow_type<T, 2> result{};
    for (const auto &elem : tens) {
        result += elem * elem;
    }
    return root<2>(result);
}
template <fixed_tensor T>
    requires(tensor_shape<T, T::size()>)
typename T::value_type norm(const T &tens) {
    return vector_norm(tens);
}
template <dynamic_tensor T> typename T::value_type norm(const T &tens) {
    assert(tens.size() == tens.shape(0));
    return vector_norm(tens);
}
// cross
template <tensor_shape<3> U, tensor_shape<3> V> auto cross(const U &x, const V &y) {
    return tensor<mult_type<U, V>, 3>{x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
}
template <dynamic_tensor U, dynamic_tensor V> auto cross(const U &x, const V &y) {
    assert(x.size() == 3 && y.size() == 3);
    assert(x.shape(0) == 3 && y.shape(0) == 3);
    return tensor<mult_type<U, V>, dynamic_shape>{x[1].data()[0] * y[2].data()[0] - x[2].data()[0] * y[1].data()[0],
                                                  x[2].data()[0] * y[0].data()[0] - x[0].data()[0] * y[2].data()[0],
                                                  x[0].data()[0] * y[1].data()[0] - x[1].data()[0] * y[0].data()[0]};
}
// translate matrix
auto translate(tensor_shape<4, 4> auto const &transform_matrix, tensor_shape<3> auto const &translation) {
    auto result = transform_matrix.copy();
    result[0][3] += translation[0];
    result[1][3] += translation[1];
    result[2][3] += translation[2];
    return result;
}
// scale matrix
auto scale(tensor_shape<4, 4> auto const &transform_matrix, tensor_shape<3> auto const &s) {
    auto result = transform_matrix.copy();
    result[0][0] *= s[0];
    result[1][1] *= s[1];
    result[2][2] *= s[2];
    return result;
}
// look_at matrix
template <tensor_shape<3> T> auto look_at(const T &eye, const T &center, const T &up) {
    tensor<typename std::remove_const<typename T::value_type>::type, 4, 4> result;
    auto z = (eye - center);
    z /= norm(z);
    auto x = cross(up, z);
    x /= norm(x);
    auto y = cross(z, x);
    y /= norm(y);
    result[0][0] = x[0];
    result[0][1] = x[1];
    result[0][2] = x[2];
    result[0][3] = -dot(x.transpose(), eye);
    result[1][0] = y[0];
    result[1][1] = y[1];
    result[1][2] = y[2];
    result[1][3] = -dot(y.transpose(), eye);
    result[2][0] = z[0];
    result[2][1] = z[1];
    result[2][2] = z[2];
    result[2][3] = -dot(z.transpose(), eye);
    result[3][0] = 0;
    result[3][1] = 0;
    result[3][2] = 0;
    result[3][3] = 1.0F;
    return result;
}
// rotate matrix
template <tensor_shape<4, 4> U, tensor_shape<3> T>
auto rotate(const U &transform_matrix, typename T::value_type const angle, const T &axis) {
    const typename T::value_type c = std::cos(angle);
    const typename T::value_type s = std::sin(angle);

    tensor<typename T::value_type, 3> axis_normed(axis / norm(axis));
    tensor<typename T::value_type, 3> temp((typename T::value_type(1) - c) * axis_normed);

    tensor<typename T::value_type, 4, 4> rotation = tensor<typename T::value_type, 4, 4>::I();
    auto A = tensor<typename T::value_type, 3, 3>::I() * c;
    tensor<typename T::value_type, 3, 3> B{typename T::value_type(0),
                                           typename T::value_type(axis_normed[2]),
                                           -typename T::value_type(axis_normed[1]),
                                           -typename T::value_type(axis_normed[2]),
                                           typename T::value_type(0),
                                           typename T::value_type(axis_normed[0]),
                                           typename T::value_type(axis_normed[1]),
                                           -typename T::value_type(axis_normed[0]),
                                           typename T::value_type(0)};
    auto C = ((typename T::value_type(1) - c)) * (axis_normed * axis_normed.transpose());
    auto R = A + s * B + C;
    rotation.template at<3, 3>(0, 0) = R;
    return rotation * transform_matrix;
}
// orthographic projection matrix
template <typename T> tensor<T, 4, 4> ortho(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax) {
    tensor<T, 4, 4> result = tensor<T, 4, 4>::I();
    result[0][0] = T(2) / (xmax - xmin);
    result[1][1] = T(2) / (ymax - ymin);
    result[2][2] = -T(2) / (zmax - zmin);
    result[0][3] = (xmin + xmax) / (xmax - xmin);
    result[1][3] = -(ymin + ymax) / (ymax - ymin);
    result[2][3] = -(zmin + zmax) / (zmax - zmin);
    return result;
}
// perspective projection matrix
// fovy in radians
template <typename T> tensor<T, 4, 4> perspective(T fovy, T aspect, T znear, T zfar) {
    tensor<T, 4, 4> result{};
    T tan_half_fovy = std::tan(fovy / T(2));
    result[0][0] = T(1) / (aspect * tan_half_fovy);
    result[1][1] = T(1) / (tan_half_fovy);
    result[2][2] = -(zfar + znear) / (zfar - znear);
    result[3][2] = -T(1);
    result[2][3] = -(T(2) * zfar * znear) / (zfar - znear);
    return result;
}
// matrix multiplication
template <tensorial T, tensorial U, tensorial V>
// requires(!scalar<T> && !scalar<U> && !scalar<V>)
void matrix_mult(const T &x, const U &y, V &A) {
    if (x.size() == 1 && y.size() == 1) {
        A.data()[0] = x.data()[0] * y.data()[0];
    } else if (x.size() == 1) {
        auto it2 = y.begin();
        for (auto it1 = A.begin(); it2 != y.end(); it1++, it2++) {
            *it1 = x.data()[0] * (*it2);
        }
    } else if (y.size() == 1) {
        auto it2 = x.begin();
        for (auto it1 = A.begin(); it2 != x.end(); it1++, it2++) {
            *it1 = y.data()[0] * (*it2);
        }
    } else {
        int i = 0;
        int j = 0;
        for (const auto &row : x.rows()) {
            j = 0;
            for (const auto &col : y.cols()) {
                typename V::value_type s{};
                auto it2 = col.begin();
                for (auto it1 = row.begin(); it2 != col.end(); it1++, it2++) {
                    s += (*it1) * (*it2);
                }
                A.data()[i + x.shape(0) * j] = s;
                j++;
            }
            i++;
        }
    }
}
template <fixed_tensor T, fixed_tensor U>
    requires(!scalar<T> && !scalar<U> && T::shape(1) == U::shape(0))
auto operator*(const T &x, const U &y) {
    constexpr auto t = squint::remove_trailing<T::shape(0), U::shape(1)>();
    constexpr size_t N = std::tuple_size<decltype(t)>::value;
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<mult_type<T, U>, std::get<I>(t)...> A;
        matrix_mult(x, y, A);
        return A;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T, dynamic_tensor U> auto operator*(const T &x, const U &y) {
    assert(x.shape(1) == y.shape(0) || x.size() == 1 || y.size() == 1);
    std::vector<size_t> result_shape{};
    if (x.size() == 1) {
        result_shape = y.shape();
    } else if (y.size() == 1) {
        result_shape = x.shape();
    } else {
        result_shape.push_back(x.shape(0));
        if (y.shape(1) != 1) {
            result_shape.push_back(y.shape(1));
        }
    }
    tensor<mult_type<T, U>, dynamic_shape> A(result_shape, mult_type<T, U>(0));
    matrix_mult(x, y, A);
    return A;
}
// dot
template <fixed_tensor U, fixed_tensor V>
    requires(tensor_shape<U, 1, U::size()> && tensor_shape<V, U::size()>)
auto dot(const U &x, const V &y) {
    return x * y;
}
template <dynamic_tensor U, dynamic_tensor V> auto dot(const U &x, const V &y) {
    assert(x.size() == y.size());
    assert(x.shape().size() == 2);
    assert(x.shape(1) == x.size());
    assert(y.shape().size() == 1);
    return x * y;
}
// scale tensors
template <tensorial T, tensorial U>
    requires(!scalar<T> && !scalar<U>)
void tensor_scale(const T &x, const scalar auto &a, U &y) {
    auto it2 = y.begin();
    for (auto it1 = x.begin(); it2 != y.end(); it1++, it2++) {
        *it2 = *it1 * a;
    }
}
template <fixed_tensor T>
    requires(!scalar<T>)
auto operator*(const scalar auto &a, const T &x) {
    using res_type = typename std::remove_const<decltype(a * std::declval<typename T::value_type &>())>::type;
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<res_type, T::shape()[I]...> c;
        tensor_scale(x, a, c);
        return c;
    }(std::make_index_sequence<N>{});
}
template <fixed_tensor T>
    requires(!scalar<T>)
auto operator*(const T &x, const scalar auto &a) {
    using res_type = typename std::remove_const<decltype(a * std::declval<typename T::value_type &>())>::type;
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<res_type, T::shape()[I]...> c;
        tensor_scale(x, a, c);
        return c;
    }(std::make_index_sequence<N>{});
}
template <fixed_tensor T>
    requires(!scalar<T>)
auto operator/(const T &x, const scalar auto &a) {
    using res_type = typename std::remove_const<decltype(std::declval<typename T::value_type &>() / a)>::type;
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<res_type, T::shape()[I]...> c;
        auto b = 1. / a;
        tensor_scale(x, b, c);
        return c;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T, scalar S>
    requires(!tensorial<S>)
auto operator*(const S &a, const T &x) {
    using res_type = typename std::remove_const<decltype(a * std::declval<typename T::value_type &>())>::type;
    auto c = x.template copy_as<res_type>();
    tensor_scale(x, a, c);
    return c;
}
template <dynamic_tensor T, scalar S>
    requires(!tensorial<S>)
auto operator*(const T &x, const S &a) {
    using res_type = typename std::remove_const<decltype(a * std::declval<typename T::value_type &>())>::type;
    auto c = x.template copy_as<res_type>();
    tensor_scale(x, a, c);
    return c;
}
template <dynamic_tensor T, scalar S>
    requires(!tensorial<S>)
auto operator/(const T &x, const S &a) {
    using res_type = typename std::remove_const<decltype(std::declval<typename T::value_type &>() / a)>::type;
    auto c = x.template copy_as<res_type>();
    auto b = 1. / a;
    tensor_scale(x, b, c);
    return c;
}
template <dynamic_tensor T, scalar S>
    requires(tensorial<S>)
auto operator*(const S &a, const T &x) {
    using res_type = typename std::remove_const<decltype(a.data()[0] * std::declval<typename T::value_type &>())>::type;
    auto c = x.template copy_as<res_type>();
    tensor_scale(x, a.data()[0], c);
    return c;
}
template <dynamic_tensor T, scalar S>
    requires(tensorial<S>)
auto operator*(const T &x, const S &a) {
    using res_type = typename std::remove_const<decltype(a.data()[0] * std::declval<typename T::value_type &>())>::type;
    auto c = x.template copy_as<res_type>();
    tensor_scale(x, a.data()[0], c);
    return c;
}
template <dynamic_tensor T, scalar S>
    requires(tensorial<S>)
auto operator/(const T &x, const S &a) {
    using res_type = typename std::remove_const<decltype(std::declval<typename T::value_type &>() / a.data()[0])>::type;
    auto c = x.template copy_as<res_type>();
    auto b = 1. / a.data()[0];
    tensor_scale(x, b, c);
    return c;
}

// solve square linear systems
template <tensorial T, tensorial V> void tensor_solve(T &A, V &B) {
    int N = A.shape(0);
    // PA=LU factorization (Doolittle algorithm)
    auto U = A.copy();
    auto L = A.template copy_as<div_type<T, T>>();
    std::fill(L.begin(), L.end(), div_type<T, T>());
    auto P = L;
    for (int i = 0; i < N; i++) {
        P[i][i] = div_type<T, T>(1.0);
    }
    for (int k = 0; k < N - 1; k++) {
        // select index >= k to maximize |A[i][k]|
        int index = k;
        auto val = typename T::value_type();
        for (int i = k; i < N; i++) {
            auto a_val = A[i][k];
            a_val = a_val < typename T::value_type(0) ? -a_val : a_val;
            if (a_val > val) {
                index = i;
                val = a_val;
            }
        }
        // Swap Rows
        auto U_row = U[k].copy();
        U[k] = U[index];
        U[index] = U_row;
        auto L_row = L[k].copy();
        L[k] = L[index];
        L[index] = L_row;
        auto P_row = P[k].copy();
        P[k] = P[index];
        P[index] = P_row;
        // compute factorization
        for (int j = k + 1; j < N; j++) {
            L[j][k] = U[j][k] / U[k][k];
            for (int i = k; i < N; i++) {
                U[j][i] = U[j][i] - L[j][k] * U[k][i];
            }
        }
    }
    // fill diagonals of L with 1
    for (int i = 0; i < N; i++) {
        L[i][i] = div_type<T, T>(1.0);
    }
    // for each column in B, solve the system using forward and back substitution
    for (auto &col : B.cols()) {
        // forward substitute
        auto y = col.template copy_as<mult_type<T, V>>();
        auto b = col.template copy_as<mult_type<T, V>>();
        // permute b
        b = P * b;
        for (int i = 0; i < N; i++) {
            auto tmp = b[i];
            for (int j = 0; j < i; j++) {
                tmp -= L[i][j] * y[j];
            }
            y[i] = tmp / L[i][i];
        }
        // back substitute into column
        for (int i = N - 1; i > -1; i--) {
            auto tmp = y[i];
            for (int j = i + 1; j < N; j++) {
                tmp -= U[i][j] * col[j];
            }
            col[i] = tmp / U[i][i];
        }
    }
}
template <fixed_tensor T, fixed_tensor U>
    requires(!scalar<T> && !scalar<U> && T::shape(0) == T::shape(1) && T::shape(1) == U::shape(0))
auto solve(const T &A, const U &B) {
    auto X = B.template copy_as<div_type<U, T>>();
    auto A_copy = A.copy();
    tensor_solve(A_copy, X);
    return X;
}
template <dynamic_tensor T, dynamic_tensor U> auto solve(const T &A, const U &B) {
    assert(A.shape(1) == B.shape(0) && A.shape(0) == A.shape(1) && A.size() > 1 && B.size() > 1);
    auto X = B.template copy_as<div_type<U, T>>();
    auto A_copy = A.copy();
    tensor_solve(A_copy, X);
    return X;
}
// solve linear least squares for over and underdetermined systems
template <fixed_tensor T, fixed_tensor U, fixed_tensor V> void tensor_solve_lls(T &A, U &B, V &X) {
    constexpr size_t M = T::shape(0);
    constexpr size_t N = T::shape(1);
    // solve underdetermined case
    if (M < N) {
        // QR factorization of transpose of A
        tensor<typename T::value_type, M, M> R_init(
            typename T::value_type{}); // same dimension as trans A, different shape
        auto R = R_init.remove_trailing();
        tensor<div_type<T, T>, N, M> Q_init(div_type<T, T>{}); // dimensionless, same shape as trans A
        auto Q = Q_init.remove_trailing();
        auto A_trans = A.transpose();
        int k;
        int i;
        int j;
        for (k = 0; k < M; k++) {
            mult_type<T, T> r(0);
            for (i = 0; i < N; i++) {
                r += A_trans[i][k] * A_trans[i][k];
            }
            R[k][k] = root<2>(r);
            for (i = 0; i < N; i++) {
                Q[i][k] = A_trans[i][k] / R[k][k];
            }
            for (j = k + 1; j < M; j++) {
                R[k][j] = typename T::value_type(0);
                for (i = 0; i < N; i++) {
                    R[k][j] += Q[i][k] * A_trans[i][j];
                }

                for (i = 0; i < N; i++) {
                    A_trans[i][j] -= R[k][j] * Q[i][k];
                }
            }
        }
        // solve for each column in B
        auto R_trans = R.transpose();
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x = x_cols.begin();
        if (B.size() == 1) {
            // forward substitute
            typename V::value_type y;
            auto tmp = static_cast<mult_type<T, V>>(B.data()[0]);
            y = tmp / R_trans[0][0];
            for (int i = 0; i < X.size(); i++) {
                X.data()[i] = Q.data()[i] * y;
            }
        } else {
            for (auto b = b_cols.begin(); x != x_cols.end(); x++, b++) {
                // forward substitute
                tensor<typename V::value_type, U::shape(0)> y_init;
                auto y = y_init.remove_trailing();
                for (int i = 0; i < U::shape(0); i++) {
                    auto tmp = static_cast<mult_type<T, V>>((*b)[i]);
                    for (int j = 0; j < i; j++) {
                        tmp -= R_trans[i][j] * static_cast<typename V::value_type>(y[j]);
                    }
                    y[i] = tmp / R_trans[i][i];
                }
                (*x) = Q * y;
            }
        }
    } else {
        // solve over determined system
        // QR factorization of A
        tensor<typename T::value_type, N, N> R_init(
            typename T::value_type{}); // same dimension as trans A, different shape
        auto R = R_init.remove_trailing();
        tensor<div_type<T, T>, M, N> Q_init(div_type<T, T>{}); // dimensionless, same shape as trans A
        auto Q = Q_init.remove_trailing();
        int k;
        int i;
        int j;
        for (k = 0; k < N; k++) {
            mult_type<T, T> r(0);
            for (i = 0; i < M; i++) {
                r += A[i][k] * A[i][k];
            }
            R[k][k] = root<2>(r);
            for (i = 0; i < M; i++) {
                Q[i][k] = A[i][k] / R[k][k];
            }
            for (j = k + 1; j < N; j++) {
                R[k][j] = typename T::value_type(0);
                for (i = 0; i < M; i++) {
                    R[k][j] += Q[i][k] * A[i][j];
                }

                for (i = 0; i < M; i++) {
                    A[i][j] -= R[k][j] * Q[i][k];
                }
            }
        }
        // solve for each column in B
        auto Q_trans = Q.transpose();
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x = x_cols.begin();
        if (X.size() == 1) {
            auto y = Q_trans * B;
            // back substitute into column
            auto tmp = static_cast<mult_type<T, V>>(y[0].data()[0]);
            X.data()[0] = tmp / R[0][0];
        } else {
            for (auto b = b_cols.begin(); x != x_cols.end(); x++, b++) {
                auto y = Q_trans * (*b);
                // back substitute into column
                for (int i = N - 1; i > -1; i--) {
                    auto tmp = static_cast<mult_type<T, V>>(y[i].data()[0]);
                    for (int j = i + 1; j < N; j++) {
                        tmp -= R[i][j] * static_cast<typename V::value_type>((*x)[j]);
                    }
                    (*x)[i] = tmp / R[i][i];
                }
            }
        }
    }
}
// solve linear least squares for over and underdetermined systems
template <dynamic_tensor T, dynamic_tensor U, dynamic_tensor V> void tensor_solve_lls(T &A, U &B, V &X) {
    size_t M = A.shape(0);
    size_t N = A.shape(1);
    // solve underdetermined case
    if (M < N) {
        // QR factorization of transpose of A
        tensor<typename T::value_type, dynamic_shape> R_init(
            {M, M}, typename T::value_type{}); // same dimension as trans A, different shape
        auto R = R_init.remove_trailing();
        tensor<div_type<T, T>, dynamic_shape> Q_init({N, M}, div_type<T, T>{}); // dimensionless, same shape as trans A
        auto Q = Q_init.remove_trailing();
        auto A_trans = A.transpose();
        int k;
        int i;
        int j;
        for (k = 0; k < M; k++) {
            mult_type<T, T> r(0);
            for (i = 0; i < N; i++) {
                r += A_trans[i][k] * A_trans[i][k];
            }
            R[k][k] = root<2>(r);
            for (i = 0; i < N; i++) {
                Q[i][k] = A_trans[i][k] / R[k][k];
            }
            for (j = k + 1; j < M; j++) {
                R[k][j] = typename T::value_type(0);
                for (i = 0; i < N; i++) {
                    R[k][j] += Q[i][k] * A_trans[i][j];
                }

                for (i = 0; i < N; i++) {
                    A_trans[i][j] -= R[k][j] * Q[i][k];
                }
            }
        }
        // solve for each column in B
        auto R_trans = R.transpose();
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x = x_cols.begin();
        for (auto b = b_cols.begin(); x != x_cols.end(); x++, b++) {
            // forward substitute
            tensor<typename V::value_type, dynamic_shape> y_init({B.shape(0)}, typename V::value_type(0));
            auto y = y_init.remove_trailing();
            for (int i = 0; i < B.shape(0); i++) {
                auto tmp = static_cast<mult_type<T, V>>((*b)[i]);
                for (int j = 0; j < i; j++) {
                    tmp -= R_trans[i][j] * static_cast<typename V::value_type>(y[j]);
                }
                y[i] = tmp / R_trans[i][i];
            }
            (*x) = Q * y;
        }
    } else {
        // solve over determined system
        // QR factorization of A
        tensor<typename T::value_type, dynamic_shape> R_init(
            {N, N}, typename T::value_type{}); // same dimension as trans A, different shape
        auto R = R_init.remove_trailing();
        tensor<div_type<T, T>, dynamic_shape> Q_init({M, N}, div_type<T, T>{}); // dimensionless, same shape as trans A
        auto Q = Q_init.remove_trailing();
        int k;
        int i;
        int j;
        for (k = 0; k < N; k++) {
            mult_type<T, T> r(0);
            for (i = 0; i < M; i++) {
                r += A[i][k] * A[i][k];
            }
            R[k][k] = root<2>(r);
            for (i = 0; i < M; i++) {
                Q[i][k] = A[i][k] / R[k][k];
            }
            for (j = k + 1; j < N; j++) {
                R[k][j] = typename T::value_type(0);
                for (i = 0; i < M; i++) {
                    R[k][j] += Q[i][k] * A[i][j];
                }

                for (i = 0; i < M; i++) {
                    A[i][j] -= R[k][j] * Q[i][k];
                }
            }
        }
        // solve for each column in B
        auto Q_trans = Q.transpose();
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x = x_cols.begin();
        for (auto b = b_cols.begin(); x != x_cols.end(); x++, b++) {
            auto y = Q_trans * (*b);
            // back substitute into column
            for (int i = N - 1; i > -1; i--) {
                auto tmp = static_cast<mult_type<T, V>>(y[i]);
                for (int j = i + 1; j < N; j++) {
                    tmp -= R[i][j] * static_cast<typename V::value_type>((*x)[j]);
                }
                (*x)[i] = tmp / R[i][i];
            }
        }
    }
}
template <fixed_tensor T, fixed_tensor U>
    requires(!scalar<T> && T::shape(0) == U::shape(0) && T::shape().size() <= 2 && U::shape().size() <= 2)
auto solve_lls(const T &A, const U &B) {
    constexpr auto t = squint::remove_trailing<T::shape(1), U::shape(1)>();
    constexpr size_t N = std::tuple_size<decltype(t)>::value;
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<div_type<U, T>, std::get<I>(t)...> X;
        auto A_copy = A.copy();
        auto B_copy = B.copy();
        tensor_solve_lls(A_copy, B_copy, X);
        return X;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T, dynamic_tensor U> auto solve_lls(const T &A, const U &B) {
    assert(A.shape(0) == B.shape(0) && A.size() > 1);
    assert(A.shape(0) * A.shape(1) == A.size());
    assert(B.shape(0) * B.shape(1) == B.size());
    std::vector<size_t> x_shape{A.shape(1)};
    if (B.shape(1) > 1) {
        x_shape.push_back(B.shape(1));
    }
    tensor<div_type<U, T>, dynamic_shape> X(x_shape, div_type<U, T>{});
    auto A_copy = A.copy();
    auto B_copy = B.copy();
    tensor_solve_lls(A_copy, B_copy, X);
    return X;
}
// general matrix division
template <fixed_tensor T, fixed_tensor U>
    requires(!scalar<T> && T::shape(0) == U::shape(0) && T::shape().size() <= 2 && U::shape().size() <= 2)
auto operator/(const U &B, const T &A) {
    constexpr auto t = squint::remove_trailing<T::shape(1), U::shape(1)>();
    constexpr size_t N = std::tuple_size<decltype(t)>::value;
    return [&]<auto... I>(std::index_sequence<I...>) {
        tensor<div_type<U, T>, std::get<I>(t)...> X;
        T A_copy(A);
        U B_copy(B);
        tensor_solve_lls(A_copy, B_copy, X);
        return X;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T, dynamic_tensor U> auto operator/(const U &B, const T &A) {
    assert(A.shape(0) == B.shape(0));
    if (A.size() == 1) {
        tensor<div_type<U, T>, dynamic_shape> X(B.shape(), div_type<U, T>{});
        auto b = 1. / (*A.data());
        tensor_scale(B, b, X);
        return X;
    }
    assert(A.shape(0) * A.shape(1) == A.size());
    assert(B.shape(0) * B.shape(1) == B.size());
    std::vector<size_t> x_shape{A.shape(1)};
    if (B.shape(1) > 1) {
        x_shape.push_back(B.shape(1));
    }
    tensor<div_type<U, T>, dynamic_shape> X(x_shape, div_type<U, T>{});
    auto A_copy = A.copy();
    auto B_copy = B.copy();
    tensor_solve_lls(A_copy, B_copy, X);
    return X;
}
// inverses
template <fixed_tensor T, fixed_tensor U> void tensor_inv(const T &A, U &A_inv) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... J>(std::index_sequence<J...>) {
        auto I = tensor<div_type<T, T>, T::shape()[J]...>::I(); // identity matrix
        A_inv = solve(A, I);
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T, dynamic_tensor U> void tensor_inv(const T &A, U &A_inv) {
    auto I = tensor<div_type<T, T>, dynamic_shape>::I(A.shape(0)); // identity matrix
    A_inv = solve(A, I);
}
template <fixed_tensor T>
    requires(!scalar<T> && T::shape(0) == T::shape(1) && T::order() == 2)
auto inv(const T &A) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... J>(std::index_sequence<J...>) {
        auto A_inv = tensor<div_type<tensor<div_type<T, T>>, T>, T::shape()[J]...>();
        tensor_inv(A, A_inv);
        return A_inv;
    }(std::make_index_sequence<N>{});
}
template <dynamic_tensor T> auto inv(const T &A) {
    assert(A.shape(0) == A.shape(1) && A.size() > 1 && A.order() == 2);
    auto A_inv = tensor<div_type<tensor<div_type<T, T>>, T>, dynamic_shape>();
    tensor_inv(A, A_inv);
    return A_inv;
}
// pinv
template <fixed_tensor T> auto pinv(const T &A) {
    tensor<div_type<tensor<div_type<T, T>>, T>, T::shape(1), T::shape(0)> result;
    auto A_l = A.transpose() * A;
    try {
        auto A_l_inv = inv(A_l); // can throw if A_l is singular
        result = A_l_inv * A.transpose();
    } catch (std::exception &e) {
        // use the right-hand-side formula
        auto A_r = A * A.transpose();
        auto A_r_inv = inv(A_r); // this should not throw if A_l was singular
        result = A.transpose() * A_r_inv;
    }
    return result;
}
template <dynamic_tensor T> auto pinv(const T &A) {
    assert(A.shape(0) * A.shape(1) == A.size() && A.size() > 1);
    tensor<div_type<tensor<div_type<T, T>>, T>, dynamic_shape> result({A.shape(1), A.shape(0)},
                                                                      div_type<tensor<div_type<T, T>>, T>(0));
    auto A_l = A.transpose() * A;
    try {
        auto A_l_inv = inv(A_l); // can throw if A_l is singular
        result = A_l_inv * A.transpose();
    } catch (std::exception &e) {
        // use the right-hand-side formula
        auto A_r = A * A.transpose();
        auto A_r_inv = inv(A_r); // this should not throw if A_l was singular
        result = A.transpose() * A_r_inv;
    }
    return result;
}
// Use BLAS and LAPACK overrides if compiled with MKL
#ifdef SQUINT_USE_MKL
// BLAS ----------------------------------------------------------------------------------------------------------------
// norm
template <typename T> T blas_norm(const MKL_INT *size, const T *data, const MKL_INT *stride);
template <> float blas_norm(const MKL_INT *size, const float *data, const MKL_INT *stride) {
    return snrm2(size, data, stride);
}
template <> double blas_norm(const MKL_INT *size, const double *data, const MKL_INT *stride) {
    return dnrm2(size, data, stride);
}
template <tensorial T>
    requires(tensor_underlying_type<T, float> || tensor_underlying_type<T, double>)
auto vector_norm(const T &tens) {
    using blas_type = typename underlying_type<T>::type;
    MKL_INT stride = tens.strides()[0];
    MKL_INT size = tens.size();
    return typename T::value_type(blas_norm(&size, (blas_type *)tens.data(), &stride));
}
// dot
template <typename T>
T blas_dot(const MKL_INT size, const T *data_x, const MKL_INT stride_x, const T *data_y,
                  const MKL_INT stride_y);
template <>
float blas_dot(const MKL_INT size, const float *data_x, const MKL_INT stride_x, const float *data_y,
               const MKL_INT stride_y) {
    return sdot(&size, data_x, &stride_x, data_y, &stride_y);
}
template <>
double blas_dot(const MKL_INT size, const double *data_x, const MKL_INT stride_x, const double *data_y,
                const MKL_INT stride_y) {
    return ddot(&size, data_x, &stride_x, data_y, &stride_y);
}
template <fixed_tensor U, fixed_tensor V>
    requires(tensor_shape<U, 1, U::size()> && tensor_shape<V, U::size()> &&
             (tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
              tensor_underlying_type<U, double> && tensor_underlying_type<V, double>))
void matrix_mult(const U &x, const V &y, mult_type<U, V> &A) {
    using blas_type = typename underlying_type<U>::type;
    MKL_INT stride_x = x.strides()[1];
    MKL_INT stride_y = y.strides()[0];
    MKL_INT size = x.size();
    A = mult_type<U, V>(blas_dot(size, (blas_type *)x.data(), stride_x, (blas_type *)y.data(), stride_y));
}
// outer product column and row vector
template <typename T>
void blas_ger(const MKL_INT M, const MKL_INT N, const T *x_data, MKL_INT stride_x, const T *y_data,
                     MKL_INT stride_y, T *A_data);
template <>
void blas_ger(const MKL_INT M, const MKL_INT N, const double *x_data, MKL_INT stride_x, const double *y_data,
              MKL_INT stride_y, double *A_data) {
    double alpha = 1.0;
    dger(&M, &N, &alpha, x_data, &stride_x, y_data, &stride_y, A_data, &M);
}
template <>
void blas_ger(const MKL_INT M, const MKL_INT N, const float *x_data, MKL_INT stride_x, const float *y_data,
              MKL_INT stride_y, float *A_data) {
    float alpha = 1.0;
    sger(&M, &N, &alpha, x_data, &stride_x, y_data, &stride_y, A_data, &M);
}
template <fixed_tensor U, fixed_tensor V>
    requires(tensor_shape<U, U::size()> && tensor_shape<V, 1, U::size()> &&
             (tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
              tensor_underlying_type<U, double> && tensor_underlying_type<V, double>))
void matrix_mult(const U &x, const V &y, tensor<mult_type<U, V>, U::shape(0), V::shape(1)> &A) {
    using blas_type = typename underlying_type<U>::type;
    const MKL_INT M = U::shape(0);
    const MKL_INT N = V::shape(1);
    double alpha = 1.0;
    MKL_INT incx = x.strides()[0];
    MKL_INT incy = y.strides()[1];
    blas_ger(M, N, (blas_type *)x.data(), incx, (blas_type *)y.data(), incy, (blas_type *)A.data());
}
// matrix-vector product
template <typename T>
void blas_gemv(const char trans_A, const MKL_INT M, const MKL_INT N, const T *A_data, MKL_INT LDA,
                      const T *x_data, const MKL_INT stride_x, T *y_data, MKL_INT stride_y);
template <>
void blas_gemv(const char trans_A, const MKL_INT M, const MKL_INT N, const double *A_data, MKL_INT LDA,
               const double *x_data, const MKL_INT stride_x, double *y_data, MKL_INT stride_y) {
    double alpha = 1.0;
    double beta = 0.0;
    dgemv(&trans_A, &M, &N, &alpha, A_data, &LDA, x_data, &stride_x, &beta, y_data, &stride_y);
}
template <>
void blas_gemv(const char trans_A, const MKL_INT M, const MKL_INT N, const float *A_data, MKL_INT LDA,
               const float *x_data, const MKL_INT stride_x, float *y_data, MKL_INT stride_y) {
    float alpha = 1.0;
    float beta = 0.0;
    sgemv(&trans_A, &M, &N, &alpha, A_data, &LDA, x_data, &stride_x, &beta, y_data, &stride_y);
}
template <fixed_tensor U, fixed_tensor V>
    requires(!scalar<U> && U::shape(0) * U::shape(1) == U::size() && tensor_shape<V, U::shape(0)> &&
             (tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
              tensor_underlying_type<U, double> && tensor_underlying_type<V, double>))
void matrix_mult(const U &A, const V &x, tensor<mult_type<U, V>, U::shape(0)> &y) {
    using blas_type = typename underlying_type<U>::type;
    const MKL_INT M = U::shape(0), N = U::shape(1);
    const MKL_INT incx = x.strides()[0];
    MKL_INT incy = 1;
    const MKL_INT A_d1_stride = A.strides()[0];
    const MKL_INT A_d2_stride = A.strides().size() > 1 ? A.strides()[1] : A.size();
    MKL_INT LDA = A_d2_stride;
    char trans_A = 'N';
    if (A_d1_stride != 1 && A_d2_stride == 1) {
        // A is transposed
        LDA = A_d1_stride;
        trans_A = 'T';
    } else if (A_d2_stride != 1) {
        // need to copy to a dense tensor since LAPACK does not work with strided matrices
        auto A_copy = A.copy();
        LDA = A_copy.strides()[1];
        blas_gemv(trans_A, M, N, (blas_type *)A_copy.data(), LDA, (blas_type *)x.data(), incx, (blas_type *)y.data(),
                  incy);
        return;
    }
    blas_gemv(trans_A, M, N, (blas_type *)A.data(), LDA, (blas_type *)x.data(), incx, (blas_type *)y.data(), incy);
}
// matrix-matrix product
template <typename T>
void blas_gemm(const char trans_A, const char trans_B, const MKL_INT M, const MKL_INT N, const MKL_INT K,
                      const T *A_data, MKL_INT LDA, const T *B_data, const MKL_INT LDB, T *C_data);
template <>
void blas_gemm(const char trans_A, const char trans_B, const MKL_INT M, const MKL_INT N, const MKL_INT K,
               const double *A_data, MKL_INT LDA, const double *B_data, const MKL_INT LDB, double *C_data) {
    double alpha = 1.0;
    double beta = 0.0;
    dgemm(&trans_A, &trans_B, &M, &N, &K, &alpha, A_data, &LDA, B_data, &LDB, &beta, C_data, &M);
}
template <>
void blas_gemm(const char trans_A, const char trans_B, const MKL_INT M, const MKL_INT N, const MKL_INT K,
               const float *A_data, MKL_INT LDA, const float *B_data, const MKL_INT LDB, float *C_data) {
    float alpha = 1.0;
    float beta = 0.0;
    sgemm(&trans_A, &trans_B, &M, &N, &K, &alpha, A_data, &LDA, B_data, &LDB, &beta, C_data, &M);
}
template <fixed_tensor U, fixed_tensor V>
    requires(U::shape(0) * U::shape(1) == U::size() && V::shape(0) * V::shape(1) == V::size() &&
             U::shape(1) == V::shape(0) && U::shape(1) != 1 && V::shape(0) != 1 && V::shape(1) != 1 &&
             (tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
              tensor_underlying_type<U, double> && tensor_underlying_type<V, double>))
void matrix_mult(const U &A, const V &B, tensor<mult_type<U, V>, U::shape(0), V::shape(1)> &C) {
    using blas_type = typename underlying_type<U>::type;
    const MKL_INT M = U::shape(0), N = V::shape(1), K = V::shape(0);
    double alpha = 1.0;
    const MKL_INT A_d1_stride = A.strides()[0];
    const MKL_INT A_d2_stride = A.strides().size() > 1 ? A.strides()[1] : A.size();
    MKL_INT LDA = A_d2_stride;
    char trans_A = 'N';
    const MKL_INT B_d1_stride = B.strides()[0];
    const MKL_INT B_d2_stride = B.strides().size() > 1 ? B.strides()[1] : B.size();
    MKL_INT LDB = B_d2_stride;
    double beta = 0.0;
    char trans_B = 'N';
    if (A_d1_stride != 1 && A_d2_stride == 1) {
        // A is transposed
        LDA = A_d1_stride;
        trans_A = 'T';
    } else if (A_d1_stride != 1 && A_d2_stride != 1) {
        // A is sparse
        auto A_copy = A.copy();
        LDA = A_copy.strides()[1];
        if (B_d1_stride != 1 && B_d2_stride == 1) {
            // B is transposed
            LDB = B_d1_stride;
            trans_B = 'T';
        } else if (B_d2_stride != 1) {
            // B is sparse
            auto B_copy = B.copy();
            LDB = B_copy.strides()[1];
            blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A_copy.data(), LDA, (blas_type *)B_copy.data(), LDB,
                      (blas_type *)C.data());
            return;
        }
        blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A_copy.data(), LDA, (blas_type *)B.data(), LDB,
                  (blas_type *)C.data());
        return;
    }
    if (B_d1_stride != 1 && B_d2_stride == 1) {
        // B is transposed
        LDB = B_d1_stride;
        trans_B = 'T';
    } else if (B_d1_stride != 1 && B_d2_stride != 1) {
        // B is sparse
        auto B_copy = B.copy();
        LDB = B_copy.strides()[1];
        blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A.data(), LDA, (blas_type *)B_copy.data(), LDB,
                  (blas_type *)C.data());
        return;
    }
    blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A.data(), LDA, (blas_type *)B.data(), LDB, (blas_type *)C.data());
}
template <dynamic_tensor U, dynamic_tensor V>
    requires((tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
              tensor_underlying_type<U, double> && tensor_underlying_type<V, double>))
void matrix_mult(const U &a, const V &b, tensor<mult_type<U, V>, dynamic_shape> &y) {
    using blas_type = typename underlying_type<U>::type;
    assert(a.shape().size() <= 2 && b.shape().size() <= 2);
    assert(a.size() == 1 || b.size() == 1 || a.shape(1) == b.shape(0));
    if (a.size() == 1) {
        // scalar multiplication
        tensor<typename U::value_type> s(a.data()[0]);
        tensor_scale(b, s, y);
    } else if (b.size() == 1) {
        // scalar multiplication
        tensor<typename V::value_type> s(b.data()[0]);
        tensor_scale(a, s, y);
    } else if (a.shape(0) == 1 && b.shape(1) == 1) {
        // inner product
        assert(a.size() == b.size());
        MKL_INT size = a.size();
        MKL_INT a_stride = a.strides()[1];
        MKL_INT b_stride = b.strides()[0];
        y.data()[0] = mult_type<U, V>(blas_dot(size, (blas_type *)a.data(), a_stride, (blas_type *)b.data(), b_stride));
    } else if (a.shape(1) == 1 && b.shape(0) == 1) {
        // outer product
        assert(a.size() == b.size());
        const MKL_INT M = a.shape(0);
        const MKL_INT N = b.shape(1);
        MKL_INT incx = a.strides()[0];
        MKL_INT incy = b.strides()[1];
        blas_ger(M, N, (blas_type *)a.data(), incx, (blas_type *)b.data(), incy, (blas_type *)y.data());
    } else if (b.shape().size() == 1) {
        // matrix vector product
        size_t m = a.shape(0);
        const MKL_INT M = a.shape(0), N = a.shape(1);
        const MKL_INT incx = b.strides()[0];
        MKL_INT incy = 1;
        const MKL_INT A_d1_stride = a.strides()[0];
        const MKL_INT A_d2_stride = a.strides().size() > 1 ? a.strides()[1] : a.size();
        MKL_INT LDA = A_d2_stride;
        char trans_A = 'N';
        if (A_d1_stride != 1 && A_d2_stride == 1) {
            // A is transposed
            LDA = A_d1_stride;
            trans_A = 'T';
        } else if (A_d2_stride != 1) {
            // need to copy to a dense tensor since LAPACK does not work with strided matrices
            auto A_copy = a.copy();
            LDA = A_copy.strides()[1];
            blas_gemv(trans_A, M, N, (blas_type *)A_copy.data(), LDA, (blas_type *)b.data(), incx,
                      (blas_type *)y.data(), incy);
            return;
        }
        blas_gemv(trans_A, M, N, (blas_type *)a.data(), LDA, (blas_type *)b.data(), incx, (blas_type *)y.data(), incy);
    } else {
        assert(a.shape(1) == b.shape(0));
        const MKL_INT M = a.shape(0), N = a.shape(1), K = b.shape(1);
        const int A_d1_stride = a.strides()[0];
        const int A_d2_stride = a.strides().size() > 1 ? a.strides()[1] : a.size();
        MKL_INT LDA = A_d2_stride;
        char trans_A = 'N';
        const int B_d1_stride = b.strides()[0];
        const int B_d2_stride = b.strides().size() > 1 ? b.strides()[1] : b.size();
        MKL_INT LDB = B_d2_stride;
        char trans_B = 'N';

        if (A_d1_stride != 1 && A_d2_stride == 1) {
            // A is transposed
            LDA = A_d1_stride;
            trans_A = 'T';
        } else if (A_d1_stride != 1 && A_d2_stride != 1) {
            // A is sparse
            auto A_copy = a.copy();
            LDA = A_copy.strides()[1];
            if (B_d1_stride != 1 && B_d2_stride == 1) {
                // B is transposed
                LDB = B_d1_stride;
                trans_B = 'T';
            } else if (B_d2_stride != 1) {
                // B is sparse
                auto B_copy = b.copy();
                LDB = B_copy.strides()[1];
                blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A_copy.data(), LDA, (blas_type *)B_copy.data(), LDB,
                          (blas_type *)y.data());
                return;
            }
            blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)A_copy.data(), LDA, (blas_type *)b.data(), LDB,
                      (blas_type *)y.data());
        }
        if (B_d1_stride != 1 && B_d2_stride == 1) {
            // B is transposed
            LDB = B_d1_stride;
            trans_B = 'T';
        } else if (B_d1_stride != 1 && B_d2_stride != 1) {
            // B is sparse
            auto B_copy = b.copy();
            LDB = B_copy.strides()[1];
            blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)a.data(), LDA, (blas_type *)B_copy.data(), LDB,
                      (blas_type *)y.data());
            return;
        }
        blas_gemm(trans_A, trans_B, M, N, K, (blas_type *)a.data(), LDA, (blas_type *)b.data(), LDB,
                  (blas_type *)y.data());
    }
}
// LAPACK --------------------------------------------------------------------------------------------------------------
// solve linear systems
// A will be overwritten with LU factorization and B will be overwritten with solution vector X
// B can have one or more columns, in which case the system is solved once for each column.
// returns zero on success
template <typename T> int lapack_gesv(MKL_INT N, MKL_INT nrhs, T *A_data, MKL_INT LDA, T *B_data);
template <> int lapack_gesv(MKL_INT N, MKL_INT nrhs, double *A_data, MKL_INT LDA, double *B_data) {
    std::vector<MKL_INT> ipiv{};
    ipiv.resize(N);
    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, A_data, LDA, ipiv.data(), B_data, LDA);
    return info;
}
template <> int lapack_gesv(MKL_INT N, MKL_INT nrhs, float *A_data, MKL_INT LDA, float *B_data) {
    std::vector<MKL_INT> ipiv{};
    ipiv.resize(N);
    int info = LAPACKE_sgesv(LAPACK_COL_MAJOR, N, nrhs, A_data, LDA, ipiv.data(), B_data, LDA);
    return info;
}
template <tensorial T, tensorial U>
    requires(tensor_underlying_type<T, float> && tensor_underlying_type<U, float> ||
             tensor_underlying_type<T, double> && tensor_underlying_type<U, double>)
void tensor_solve(T &A, U &B) {
    using blas_type = typename underlying_type<T>::type;
    auto result = lapack_gesv(A.shape(0), B.shape(1), (blas_type *)A.data(), A.strides()[1], (blas_type *)B.data());
    if (result != 0) {
        throw std::runtime_error("Singular matrix");
    }
}
// matrix inverse
template <typename T> int lapack_getrf(MKL_INT N, T *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv);
template <> int lapack_getrf(MKL_INT N, double *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv) {
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, A_data, LDA, ipiv.data());
    return info;
}
template <> int lapack_getrf(MKL_INT N, float *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv) {
    int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, N, N, A_data, LDA, ipiv.data());
    return info;
}
template <typename T> int lapack_getri(MKL_INT N, T *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv);
template <> int lapack_getri(MKL_INT N, double *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv) {
    int info = LAPACKE_dgetri(LAPACK_COL_MAJOR, N, A_data, LDA, ipiv.data());
    return info;
}
template <> int lapack_getri(MKL_INT N, float *A_data, MKL_INT LDA, std::vector<MKL_INT> &ipiv) {
    int info = LAPACKE_sgetri(LAPACK_COL_MAJOR, N, A_data, LDA, ipiv.data());
    return info;
}
template <fixed_tensor T, fixed_tensor U>
    requires((tensor_underlying_type<T, float> && tensor_underlying_type<U, float>) ||
             (tensor_underlying_type<T, double> && tensor_underlying_type<U, double>))
void tensor_inv(const T &A, U &A_inv) {
    using blas_type = typename underlying_type<T>::type;
    std::vector<MKL_INT> ipiv{};
    ipiv.resize(A.shape(0));
    A_inv = A.template copy_as<div_type<tensor<div_type<T, T>>, T>>();
    auto result = lapack_getrf(A.shape(0), (blas_type *)A_inv.data(), A.strides()[1], ipiv);
    if (result != 0) {
        throw std::runtime_error("Singular matrix");
    }
    result = lapack_getri(A.shape(0), (blas_type *)A_inv.data(), A.strides()[1], ipiv);
    if (result != 0) {
        throw std::runtime_error("Singular matrix");
    }
}
template <dynamic_tensor T, dynamic_tensor U>
    requires((tensor_underlying_type<T, float> && tensor_underlying_type<U, float>) ||
             (tensor_underlying_type<T, double> && tensor_underlying_type<U, double>))
void tensor_inv(const T &A, U &A_inv) {
    using blas_type = typename underlying_type<T>::type;
    std::vector<MKL_INT> ipiv{};
    ipiv.resize(A.shape(0));
    A_inv = A.template copy_as<div_type<tensor<div_type<T, T>>, T>>();
    auto result = lapack_getrf(A.shape(0), (blas_type *)A_inv.data(), A.strides()[1], ipiv);
    if (result != 0) {
        throw std::runtime_error("Singular matrix");
    }
    result = lapack_getri(A.shape(0), (blas_type *)A_inv.data(), A.strides()[1], ipiv);
    if (result != 0) {
        throw std::runtime_error("Singular matrix");
    }
}

// solve overdetermined or underdetermiend linear systems in a least squares
// sense overwrites A with QR or LQ factorization overwrites b with solution
// vector returns 0 on success
template <typename T>
int lapack_gels(MKL_INT M, MKL_INT N, MKL_INT nrhs, T *A_data, MKL_INT LDA, T *B_data, MKL_INT LDB);
template <>
int lapack_gels(MKL_INT M, MKL_INT N, MKL_INT nrhs, double *A_data, MKL_INT LDA, double *B_data,
                MKL_INT LDB) {
    int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', M, N, nrhs, A_data, LDA, B_data, LDB);
    return info;
}
template <>
int lapack_gels(MKL_INT M, MKL_INT N, MKL_INT nrhs, float *A_data, MKL_INT LDA, float *B_data, MKL_INT LDB) {
    int info = LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', M, N, nrhs, A_data, LDA, B_data, LDB);
    return info;
}
template <fixed_tensor T, fixed_tensor U, fixed_tensor V>
    requires(tensor_underlying_type<T, float> && tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
             tensor_underlying_type<T, double> && tensor_underlying_type<U, double> &&
                 tensor_underlying_type<V, double>)
void tensor_solve_lls(T &A, U &B, V &X) {
    using blas_type = typename underlying_type<T>::type;
    MKL_INT LDA = A.shape(0);
    MKL_INT LDB = maximum(A.shape(0), A.shape(1));
    if (LDB == B.shape(0)) {
        auto result =
            lapack_gels(A.shape(0), A.shape(1), B.shape(1), (blas_type *)A.data(), LDA, (blas_type *)B.data(), LDB);
        if (result != 0) {
            throw std::runtime_error("Singular matrix");
        }
        // copy results into X
        if (X.size() == 1) {
            X.data()[0] = static_cast<typename V::value_type>(B.data()[0]);
        } else {
            auto x_cols = X.cols();
            auto b_cols = B.cols();
            auto x_col_it = x_cols.begin();
            for (auto b_col_it = b_cols.begin(); x_col_it != x_cols.end(); b_col_it++, x_col_it++) {
                auto it2 = (*x_col_it).begin();
                for (auto it1 = (*b_col_it).begin(); it2 != (*x_col_it).end(); it1++, it2++) {
                    *it2 = static_cast<typename V::value_type>(*it1);
                }
            }
        }
    } else {
        // copy B into X
        if (B.size() == 1) {
            X.data()[0] = static_cast<typename V::value_type>(B.data()[0]);
        } else {
            auto x_cols = X.cols();
            auto b_cols = B.cols();
            auto x_col_it = x_cols.begin();
            for (auto b_col_it = b_cols.begin(); b_col_it != b_cols.end(); b_col_it++, x_col_it++) {
                auto it2 = (*x_col_it).begin();
                for (auto it1 = (*b_col_it).begin(); it1 != (*b_col_it).end(); it1++, it2++) {
                    *it2 = static_cast<typename V::value_type>(*it1);
                }
            }
        }
        auto result =
            lapack_gels(A.shape(0), A.shape(1), X.shape(1), (blas_type *)A.data(), LDA, (blas_type *)X.data(), LDB);
        if (result != 0) {
            throw std::runtime_error("Singular matrix");
        }
    }
}
template <dynamic_tensor T, dynamic_tensor U, dynamic_tensor V>
    requires(tensor_underlying_type<T, float> && tensor_underlying_type<U, float> && tensor_underlying_type<V, float> ||
             tensor_underlying_type<T, double> && tensor_underlying_type<U, double> &&
                 tensor_underlying_type<V, double>)
void tensor_solve_lls(T &A, U &B, V &X) {
    using blas_type = typename underlying_type<T>::type;
    MKL_INT LDA = A.shape(0);
    MKL_INT LDB = maximum(A.shape(0), A.shape(1));
    if (LDB == B.shape(0)) {
        auto result =
            lapack_gels(A.shape(0), A.shape(1), B.shape(1), (blas_type *)A.data(), LDA, (blas_type *)B.data(), LDB);
        if (result != 0) {
            throw std::runtime_error("Singular matrix");
        }
        // copy results into X
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x_col_it = x_cols.begin();
        for (auto b_col_it = b_cols.begin(); x_col_it != x_cols.end(); b_col_it++, x_col_it++) {
            auto it2 = (*x_col_it).begin();
            for (auto it1 = (*b_col_it).begin(); it2 != (*x_col_it).end(); it1++, it2++) {
                *it2 = static_cast<typename V::value_type>(*it1);
            }
        }
    } else {
        // copy B into X
        auto x_cols = X.cols();
        auto b_cols = B.cols();
        auto x_col_it = x_cols.begin();
        for (auto b_col_it = b_cols.begin(); b_col_it != b_cols.end(); b_col_it++, x_col_it++) {
            auto it2 = (*x_col_it).begin();
            for (auto it1 = (*b_col_it).begin(); it1 != (*b_col_it).end(); it1++, it2++) {
                *it2 = static_cast<typename V::value_type>(*it1);
            }
        }
        auto result =
            lapack_gels(A.shape(0), A.shape(1), X.shape(1), (blas_type *)A.data(), LDA, (blas_type *)X.data(), LDB);
        if (result != 0) {
            throw std::runtime_error("Singular matrix");
        }
    }
}
#endif
} // namespace squint
