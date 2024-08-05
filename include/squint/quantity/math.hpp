/**
 * @file math.hpp
 * @brief Mathematical functions for quantities and arithmetic types.
 *
 * This file provides a set of mathematical functions that work with both
 * quantitative types (from the squint library) and standard arithmetic types.
 * It includes functions for comparison, basic arithmetic, trigonometry, and more.
 */

#ifndef SQUINT_QUANTITY_MATH_HPP
#define SQUINT_QUANTITY_MATH_HPP

#include "squint/core/concepts.hpp"
#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/quantity.hpp"
#include "squint/util/math_utils.hpp"

#include <cmath>

namespace squint {

/**
 * @brief Approximately equal comparison for quantities
 * @tparam T First quantitative type
 * @tparam U Second quantitative type
 * @param a First quantity
 * @param b Second quantity
 * @param epsilon Tolerance for comparison
 * @return true if quantities are approximately equal, false otherwise
 */
template <quantitative T, quantitative U>
auto approx_equal(const T &a, const U &b,
                  const typename T::value_type &epsilon = typename T::value_type{DEFAULT_EPSILON}) -> bool {
    static_assert(std::is_same_v<typename T::dimension_type, typename U::dimension_type>,
                  "Quantities must have the same dimension");
    return approx_equal(a.value(), b.value(), epsilon);
}

/**
 * @brief Approximately equal comparison for mixed types (quantitative and arithmetic)
 * @tparam T Quantitative type
 * @tparam U Arithmetic type
 * @param a Quantity
 * @param b Arithmetic value
 * @param epsilon Tolerance for comparison
 * @return true if values are approximately equal, false otherwise
 */
template <quantitative T, arithmetic U>
auto approx_equal(const T &a, const U &b, const U &epsilon = U{DEFAULT_EPSILON}) -> bool
    requires std::is_same_v<typename T::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a.value(), b, epsilon);
}

/**
 * @brief Approximately equal comparison for mixed types (arithmetic and quantitative)
 * @tparam T Arithmetic type
 * @tparam U Quantitative type
 * @param a Arithmetic value
 * @param b Quantity
 * @param epsilon Tolerance for comparison
 * @return true if values are approximately equal, false otherwise
 */
template <arithmetic T, quantitative U>
auto approx_equal(const T &a, const U &b, const T &epsilon = T{DEFAULT_EPSILON}) -> bool
    requires std::is_same_v<typename U::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a, b.value(), epsilon);
}

/**
 * @brief Absolute value function
 * @tparam T Quantitative or arithmetic type
 * @param x Input value
 * @return Absolute value of x
 */
template <typename T> auto abs(const T &x) -> T {
    if constexpr (quantitative<T>) {
        return T(std::abs(static_cast<typename T::value_type>(x)));
    } else {
        return std::abs(x);
    }
}

/**
 * @brief Square root function
 * @tparam T Quantitative or arithmetic type
 * @param x Input value
 * @return Square root of x
 */
template <typename T> auto sqrt(const T &x) {
    if constexpr (quantitative<T>) {
        using result_dimension = root_t<typename T::dimension_type, 2>;
        return quantity<decltype(std::sqrt(static_cast<typename T::value_type>(x))), result_dimension>(
            std::sqrt(static_cast<typename T::value_type>(x)));
    } else {
        return std::sqrt(x);
    }
}

/**
 * @brief Nth root function
 * @tparam T Quantitative or arithmetic type
 * @param x Input value
 * @return Nth root of x
 */
template <int N, typename T> auto root(const T &x) {
    if constexpr (quantitative<T>) {
        using result_dimension = root_t<typename T::dimension_type, N>;
        return quantity<decltype(std::pow(static_cast<typename T::value_type>(x), 1.0 / N)), result_dimension>(
            std::pow(static_cast<typename T::value_type>(x), 1.0 / N));
    } else {
        return std::pow(x, 1.0 / N);
    }
}

/**
 * @brief Exponential function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return e raised to the power of x
 */
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

/**
 * @brief Natural logarithm function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Natural logarithm of x
 */
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

/**
 * @brief Power function for quantitative types
 * @tparam N Integer power
 * @tparam T Quantitative type
 * @param x Base value
 * @return x raised to the power of N
 */
template <int N, quantitative T> auto pow(const T &x) {
    using result_dimension = pow_t<typename T::dimension_type, N>;
    return quantity<decltype(std::pow(static_cast<typename T::value_type>(x), N)), result_dimension>(
        std::pow(static_cast<typename T::value_type>(x), N));
}

/// @name Trigonometric functions
/// @{

/**
 * @brief Sine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Sine of x
 */
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

/**
 * @brief Cosine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Cosine of x
 */
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

/**
 * @brief Tangent function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Tangent of x
 */
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

/// @}

/// @name Inverse trigonometric functions
/// @{

/**
 * @brief Inverse sine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Arcsine of x
 */
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

/**
 * @brief Inverse cosine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Arccosine of x
 */
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

/**
 * @brief Inverse tangent function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Arctangent of x
 */
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

/**
 * @brief Two-argument inverse tangent function
 * @tparam T Quantitative or arithmetic type for y
 * @tparam U Quantitative or arithmetic type for x
 * @param y Y-coordinate
 * @param x X-coordinate
 * @return Arctangent of y/x, using the signs of both arguments to determine the quadrant of the return value
 */
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

/// @}

/// @name Hyperbolic functions
/// @{

/**
 * @brief Hyperbolic sine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Hyperbolic sine of x
 */
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

/**
 * @brief Hyperbolic cosine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Hyperbolic cosine of x
 */
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

/**
 * @brief Hyperbolic tangent function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Hyperbolic tangent of x
 */
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

/// @}

/// @name Inverse hyperbolic functions
/// @{

/**
 * @brief Inverse hyperbolic sine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Inverse hyperbolic sine of x
 */
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

/**
 * @brief Inverse hyperbolic cosine function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Inverse hyperbolic cosine of x
 */
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

/**
 * @brief Inverse hyperbolic tangent function (only for dimensionless quantities or arithmetic types)
 * @tparam T Quantitative (dimensionless) or arithmetic type
 * @param x Input value
 * @return Inverse hyperbolic tangent of x
 */
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

/// @}

} // namespace squint

#endif // SQUINT_QUANTITY_MATH_HPP