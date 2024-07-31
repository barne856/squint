#ifndef SQUINT_QUANTITY_QUANTITY_OPS_HPP
#define SQUINT_QUANTITY_QUANTITY_OPS_HPP

#include "squint/quantity/quantity.hpp"

namespace squint {

// Arithmetic operations between quantities
template <typename T1, typename T2, dimensional D, error_checking ErrorChecking1, error_checking ErrorChecking2>
constexpr auto operator+(const quantity<T1, D, ErrorChecking1> &lhs, const quantity<T2, D, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() + rhs.value());
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, D, result_error_checking>::check_overflow_add(lhs.value(), rhs.value());
    }

    return quantity<result_type, D, result_error_checking>(lhs.value() + rhs.value());
}

template <typename T1, typename T2, dimensional D, error_checking ErrorChecking1, error_checking ErrorChecking2>
constexpr auto operator-(const quantity<T1, D, ErrorChecking1> &lhs, const quantity<T2, D, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() - rhs.value());
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, D, result_error_checking>::check_overflow_subtract(lhs.value(), rhs.value());
    }

    return quantity<result_type, D, result_error_checking>(lhs.value() - rhs.value());
}

template <typename T1, typename T2, dimensional D1, dimensional D2, error_checking ErrorChecking1,
          error_checking ErrorChecking2>
constexpr auto operator*(const quantity<T1, D1, ErrorChecking1> &lhs, const quantity<T2, D2, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() * rhs.value());
    using result_dimension = mult_t<D1, D2>;
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_overflow_multiply(lhs.value(),
                                                                                                rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() * rhs.value());
}

template <typename T1, typename T2, dimensional D1, dimensional D2, error_checking ErrorChecking1,
          error_checking ErrorChecking2>
constexpr auto operator/(const quantity<T1, D1, ErrorChecking1> &lhs, const quantity<T2, D2, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() / rhs.value());
    using result_dimension = div_t<D1, D2>;
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_division_by_zero(rhs.value());
        quantity<result_type, result_dimension, result_error_checking>::check_underflow_divide(lhs.value(),
                                                                                               rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() / rhs.value());
}

// Scalar operations
template <arithmetic T, typename U, dimensional D, error_checking ErrorChecking>
constexpr auto operator*(const T &scalar, const quantity<U, D, ErrorChecking> &q) {
    using result_type = decltype(scalar * q.value());

    if constexpr (ErrorChecking == error_checking::enabled) {
        quantity<result_type, D, ErrorChecking>::check_overflow_multiply(scalar, q.value());
    }

    return quantity<result_type, D, ErrorChecking>(scalar * q.value());
}

template <typename T, arithmetic U, dimensional D, error_checking ErrorChecking>
constexpr auto operator*(const quantity<T, D, ErrorChecking> &q, const U &scalar) {
    return scalar * q;
}

template <typename T, arithmetic U, dimensional D, error_checking ErrorChecking>
constexpr auto operator/(const quantity<T, D, ErrorChecking> &q, const U &scalar) {
    using result_type = decltype(q.value() / scalar);

    if constexpr (ErrorChecking == error_checking::enabled) {
        quantity<result_type, D, ErrorChecking>::check_division_by_zero(scalar);
        quantity<result_type, D, ErrorChecking>::check_underflow_divide(q.value(), scalar);
    }

    return quantity<result_type, D, ErrorChecking>(q.value() / scalar);
}

template <arithmetic T, typename U, dimensional D, error_checking ErrorChecking>
constexpr auto operator/(const T &scalar, const quantity<U, D, ErrorChecking> &q) {
    using result_type = decltype(scalar / q.value());
    using result_dimension = inv_t<D>;

    if constexpr (ErrorChecking == error_checking::enabled) {
        quantity<result_type, result_dimension, ErrorChecking>::check_division_by_zero(q.value());
        quantity<result_type, result_dimension, ErrorChecking>::check_underflow_divide(scalar, q.value());
    }

    return quantity<result_type, result_dimension, ErrorChecking>(scalar / q.value());
}

// Output stream operator
template <arithmetic T, dimensional D, error_checking ErrorChecking>
std::ostream &operator<<(std::ostream &os, const quantity<T, D, ErrorChecking> &q) {
    return os << q.value();
}

// Input stream operator
template <arithmetic T, dimensional D, error_checking ErrorChecking>
std::istream &operator>>(std::istream &is, quantity<T, D, ErrorChecking> &q) {
    T value;
    is >> value;
    if constexpr (ErrorChecking == error_checking::enabled) {
        try {
            q = quantity<T, D, ErrorChecking>(value);
        } catch (const std::exception &e) {
            is.setstate(std::ios_base::failbit);
            throw;
        }
    } else {
        q = quantity<T, D, ErrorChecking>(value);
    }
    return is;
}

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_OPS_HPP