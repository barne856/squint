#ifndef SQUINT_CORE_HPP
#define SQUINT_CORE_HPP

#ifdef _WIN32
#define __declspec(empty_bases) __declspec(empty_bases)
#else
#define __declspec(empty_bases)
#endif

#include <algorithm>
#include <cassert>
#include <concepts>
#include <limits>

namespace squint {
// Define a type to enable or disable error checking
enum class error_checking { enabled, disabled };

// Layout options
enum class layout { row_major, column_major };

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

// Concept for quantities
template <typename T>
concept quantitative = requires {
    typename T::value_type;
    typename T::dimension_type;
};

// concept for scalar is arithmetic or quantitative
template <typename T>
concept scalar = arithmetic<T> || quantitative<T>;

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
#endif // SQUINT_CORE_HPP
