#ifndef SQUINT_UTIL_MATH_UTILS_HPP
#define SQUINT_UTIL_MATH_UTILS_HPP

#include "squint/core/concepts.hpp"

#include <cmath>
#include <limits>

namespace squint {

// Constexpr power function for integer exponents
template <typename T> constexpr T int_pow(T base, int exp) {
    if (exp == 0)
        return T(1);
    if (exp < 0) {
        base = T(1) / base;
        exp = -exp;
    }
    T result = T(1);
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}
// Constexpr square root function
template <typename T> constexpr T sqrt_constexpr(T x, T curr, T prev) {
    return curr == prev ? curr : sqrt_constexpr(x, (curr + x / curr) / 2, curr);
}

template <typename T> constexpr T sqrt_constexpr(T x) {
    return x >= 0 && x < std::numeric_limits<T>::infinity() ? sqrt_constexpr(x, x, T(0))
                                                            : std::numeric_limits<T>::quiet_NaN();
}

// approx equal for arithmetic types
template <typename T>
    requires arithmetic<T>
bool approx_equal(T a, T b, T epsilon = 128 * 1.192092896e-04, T abs_th = std::numeric_limits<T>::epsilon()) {
    assert(std::numeric_limits<T>::epsilon() <= epsilon);
    assert(epsilon < 1.F);

    if (a == b)
        return true;

    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
    return diff < std::max(abs_th, epsilon * norm);
}

} // namespace squint

#endif // SQUINT_UTIL_MATH_UTILS_HPP