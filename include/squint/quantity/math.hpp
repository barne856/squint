#ifndef SQUINT_QUANTITY_MATH_HPP
#define SQUINT_QUANTITY_MATH_HPP

#include "squint/core/concepts.hpp"
#include "squint/quantity/dimension.hpp"

#include <cmath>

namespace squint {

// approx equal for quantities
template <quantitative T, quantitative U, arithmetic Epsilon>
bool approx_equal(const T &a, const U &b, const Epsilon &epsilon = Epsilon{128 * 1.192092896e-04}) {
    static_assert(std::is_same_v<typename T::dimension_type, typename U::dimension_type>,
                  "Quantities must have the same dimension");
    return approx_equal(a.value(), b.value(), epsilon);
}

// approx equal for mixed types
template <quantitative T, arithmetic U, arithmetic V>
bool approx_equal(const T &a, const U &b, const V &epsilon = V{128 * 1.192092896e-04})
    requires std::is_same_v<typename T::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a.value(), b, epsilon);
}

template <arithmetic T, quantitative U, arithmetic V>
bool approx_equal(const T &a, const U &b, const V &epsilon = V{128 * 1.192092896e-04})
    requires std::is_same_v<typename U::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a, b.value(), epsilon);
}

// Absolute value
template <typename T> auto abs(const T &x) {
    if constexpr (quantitative<T>) {
        return T(std::abs(static_cast<typename T::value_type>(x)));
    } else {
        return std::abs(x);
    }
}

// Square root
template <typename T> auto sqrt(const T &x) {
    if constexpr (quantitative<T>) {
        using result_dimension = root_t<typename T::dimension_type, 2>;
        return quantity<decltype(std::sqrt(static_cast<typename T::value_type>(x))), result_dimension>(
            std::sqrt(static_cast<typename T::value_type>(x)));
    } else {
        return std::sqrt(x);
    }
}

// Exponential function (only for dimensionless quantities)
template <typename T> auto exp(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Exponential function only accepts dimensionless quantities");
        return quantity<decltype(std::exp(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::exp(static_cast<typename T::value_type>(x)));
    } else {
        return std::exp(x);
    }
}

// Natural logarithm (only for dimensionless quantities)
template <typename T> auto log(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Logarithm function only accepts dimensionless quantities");
        return quantity<decltype(std::log(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::log(static_cast<typename T::value_type>(x)));
    } else {
        return std::log(x);
    }
}

// Power function
template <int N> auto pow(quantitative auto x) {
    using result_dimension = pow_t<typename decltype(x)::dimension_type, N>;
    return quantity<decltype(std::pow(static_cast<typename decltype(x)::value_type>(x), N)), result_dimension>(
        std::pow(static_cast<typename decltype(x)::value_type>(x), N));
}

// Trigonometric functions (only for dimensionless quantities)
template <typename T> auto sin(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::sin(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::sin(static_cast<typename T::value_type>(x)));
    } else {
        return std::sin(x);
    }
}

template <typename T> auto cos(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::cos(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::cos(static_cast<typename T::value_type>(x)));
    } else {
        return std::cos(x);
    }
}

template <typename T> auto tan(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::tan(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::tan(static_cast<typename T::value_type>(x)));
    } else {
        return std::tan(x);
    }
}

// Inverse trigonometric functions (return dimensionless quantities)
template <typename T> auto asin(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::asin(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::asin(static_cast<typename T::value_type>(x)));
    } else {
        return std::asin(x);
    }
}

template <typename T> auto acos(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::acos(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::acos(static_cast<typename T::value_type>(x)));
    } else {
        return std::acos(x);
    }
}

template <typename T> auto atan(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse trigonometric functions only accept dimensionless quantities");
        return quantity<decltype(std::atan(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::atan(static_cast<typename T::value_type>(x)));
    } else {
        return std::atan(x);
    }
}

// Two-argument arctangent
template <typename T, typename U> auto atan2(const T &y, const U &x) {
    if constexpr (quantitative<T> && quantitative<U>) {
        static_assert(std::is_same_v<typename T::dimension_type, typename U::dimension_type>,
                      "atan2 arguments must have the same dimensions");
        return quantity<decltype(std::atan2(static_cast<typename T::value_type>(y),
                                            static_cast<typename U::value_type>(x))),
                        dimensions::dimensionless>(
            std::atan2(static_cast<typename T::value_type>(y), static_cast<typename U::value_type>(x)));
    } else {
        return std::atan2(quantitative<T> ? static_cast<typename T::value_type>(y) : y,
                          quantitative<U> ? static_cast<typename U::value_type>(x) : x);
    }
}

// Hyperbolic functions (only for dimensionless quantities)
template <typename T> auto sinh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::sinh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::sinh(static_cast<typename T::value_type>(x)));
    } else {
        return std::sinh(x);
    }
}

template <typename T> auto cosh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::cosh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::cosh(static_cast<typename T::value_type>(x)));
    } else {
        return std::cosh(x);
    }
}

template <typename T> auto tanh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::tanh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::tanh(static_cast<typename T::value_type>(x)));
    } else {
        return std::tanh(x);
    }
}

// Inverse hyperbolic functions (return dimensionless quantities)
template <typename T> auto asinh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::asinh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::asinh(static_cast<typename T::value_type>(x)));
    } else {
        return std::asinh(x);
    }
}

template <typename T> auto acosh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::acosh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::acosh(static_cast<typename T::value_type>(x)));
    } else {
        return std::acosh(x);
    }
}

template <typename T> auto atanh(const T &x) {
    if constexpr (quantitative<T>) {
        static_assert(std::is_same_v<typename T::dimension_type, dimensions::dimensionless>,
                      "Inverse hyperbolic functions only accept dimensionless quantities");
        return quantity<decltype(std::atanh(static_cast<typename T::value_type>(x))), dimensions::dimensionless>(
            std::atanh(static_cast<typename T::value_type>(x)));
    } else {
        return std::atanh(x);
    }
}

} // namespace squint

#endif // SQUINT_QUANTITY_MATH_HPP