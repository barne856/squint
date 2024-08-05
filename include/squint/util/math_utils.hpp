/**
 * @file math_utils.hpp
 * @brief Mathematical utility functions for the Squint library.
 *
 * This file provides mathematical utility functions, including
 * an implementation of approximate equality for floating-point types.
 */

#ifndef SQUINT_UTIL_MATH_UTILS_HPP
#define SQUINT_UTIL_MATH_UTILS_HPP

#include "squint/core/concepts.hpp"
#include <cmath>
#include <limits>

namespace squint {

/**
 * @brief Default epsilon value for floating-point comparisons.
 *
 * This value is used as the default tolerance in approximate equality comparisons.
 */
constexpr long double DEFAULT_EPSILON = 128 * 1.192092896e-04;

/**
 * @brief Checks if two arithmetic values are approximately equal.
 *
 * This function compares two values for approximate equality, taking into account
 * both relative and absolute tolerances.
 *
 * @tparam T The arithmetic type of the values being compared.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @param epsilon The relative tolerance for the comparison (default: DEFAULT_EPSILON).
 * @param abs_th The absolute tolerance for the comparison (default: std::numeric_limits<T>::epsilon()).
 * @return bool True if the values are approximately equal, false otherwise.
 *
 * @note This function requires that T satisfies the arithmetic concept.
 */
template <typename T>
    requires arithmetic<T>
auto approx_equal(T a, T b, T epsilon = DEFAULT_EPSILON, T abs_th = std::numeric_limits<T>::epsilon()) -> bool {
    assert(std::numeric_limits<T>::epsilon() <= epsilon);
    assert(epsilon < 1.F);

    if (a == b) {
        return true;
    }

    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
    return diff < std::max(abs_th, epsilon * norm);
}

} // namespace squint

#endif // SQUINT_UTIL_MATH_UTILS_HPP