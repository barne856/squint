#ifndef SQUINT_UTIL_MATH_UTILS_HPP
#define SQUINT_UTIL_MATH_UTILS_HPP

#include "squint/core/concepts.hpp"

#include <cmath>
#include <limits>

namespace squint {

/// @brief Default epsilon value for floating-point comparisons
constexpr long double DEFAULT_EPSILON = 128 * 1.192092896e-04;

// approx equal for arithmetic types
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