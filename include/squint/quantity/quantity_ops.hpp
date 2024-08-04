/**
 * @file quantity_ops.hpp
 * @brief Defines arithmetic and stream operations for quantities.
 *
 * This file provides operator overloads for arithmetic operations between
 * quantities, between quantities and scalars, and for stream I/O operations.
 */

#ifndef SQUINT_QUANTITY_QUANTITY_OPS_HPP
#define SQUINT_QUANTITY_QUANTITY_OPS_HPP

#include "squint/quantity/quantity.hpp"
#include <iostream>

namespace squint {

/**
 * @brief Addition operator for quantities.
 *
 * @tparam T1 The underlying type of the left-hand quantity.
 * @tparam T2 The underlying type of the right-hand quantity.
 * @tparam D The dimension of both quantities.
 * @tparam ErrorChecking1 Error checking policy of the left-hand quantity.
 * @tparam ErrorChecking2 Error checking policy of the right-hand quantity.
 * @param lhs Left-hand side quantity.
 * @param rhs Right-hand side quantity.
 * @return A new quantity representing the sum.
 */
template <typename T1, typename T2, dimensional D, error_checking ErrorChecking1, error_checking ErrorChecking2>
constexpr auto operator+(const quantity<T1, D, ErrorChecking1> &lhs, const quantity<T2, D, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() + rhs.value());
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, D, result_error_checking>::check_overflow_add(lhs.value(), rhs.value());
    }

    return quantity<result_type, D, result_error_checking>(lhs.value() + rhs.value());
}

/**
 * @brief Subtraction operator for quantities.
 *
 * @tparam T1 The underlying type of the left-hand quantity.
 * @tparam T2 The underlying type of the right-hand quantity.
 * @tparam D The dimension of both quantities.
 * @tparam ErrorChecking1 Error checking policy of the left-hand quantity.
 * @tparam ErrorChecking2 Error checking policy of the right-hand quantity.
 * @param lhs Left-hand side quantity.
 * @param rhs Right-hand side quantity.
 * @return A new quantity representing the difference.
 */
template <typename T1, typename T2, dimensional D, error_checking ErrorChecking1, error_checking ErrorChecking2>
constexpr auto operator-(const quantity<T1, D, ErrorChecking1> &lhs, const quantity<T2, D, ErrorChecking2> &rhs) {
    using result_type = decltype(lhs.value() - rhs.value());
    constexpr error_checking result_error_checking = resulting_error_checking<ErrorChecking1, ErrorChecking2>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, D, result_error_checking>::check_overflow_subtract(lhs.value(), rhs.value());
    }

    return quantity<result_type, D, result_error_checking>(lhs.value() - rhs.value());
}

/**
 * @brief Multiplication operator for quantities.
 *
 * @tparam T1 The underlying type of the left-hand quantity.
 * @tparam T2 The underlying type of the right-hand quantity.
 * @tparam D1 The dimension of the left-hand quantity.
 * @tparam D2 The dimension of the right-hand quantity.
 * @tparam ErrorChecking1 Error checking policy of the left-hand quantity.
 * @tparam ErrorChecking2 Error checking policy of the right-hand quantity.
 * @param lhs Left-hand side quantity.
 * @param rhs Right-hand side quantity.
 * @return A new quantity representing the product.
 */
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

/**
 * @brief Division operator for quantities.
 *
 * @tparam T1 The underlying type of the left-hand quantity.
 * @tparam T2 The underlying type of the right-hand quantity.
 * @tparam D1 The dimension of the left-hand quantity.
 * @tparam D2 The dimension of the right-hand quantity.
 * @tparam ErrorChecking1 Error checking policy of the left-hand quantity.
 * @tparam ErrorChecking2 Error checking policy of the right-hand quantity.
 * @param lhs Left-hand side quantity.
 * @param rhs Right-hand side quantity.
 * @return A new quantity representing the quotient.
 */
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

/**
 * @brief Multiplication operator for scalar and quantity.
 *
 * @tparam T The scalar type.
 * @tparam U The underlying type of the quantity.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param scalar The scalar value.
 * @param q The quantity.
 * @return A new quantity representing the product.
 */
template <arithmetic T, typename U, dimensional D, error_checking ErrorChecking>
constexpr auto operator*(const T &scalar, const quantity<U, D, ErrorChecking> &q) {
    using result_type = decltype(scalar * q.value());

    if constexpr (ErrorChecking == error_checking::enabled) {
        quantity<result_type, D, ErrorChecking>::check_overflow_multiply(scalar, q.value());
    }

    return quantity<result_type, D, ErrorChecking>(scalar * q.value());
}

/**
 * @brief Multiplication operator for quantity and scalar.
 *
 * @tparam T The underlying type of the quantity.
 * @tparam U The scalar type.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param q The quantity.
 * @param scalar The scalar value.
 * @return A new quantity representing the product.
 */
template <typename T, arithmetic U, dimensional D, error_checking ErrorChecking>
constexpr auto operator*(const quantity<T, D, ErrorChecking> &q, const U &scalar) {
    return scalar * q;
}

/**
 * @brief Division operator for quantity and scalar.
 *
 * @tparam T The underlying type of the quantity.
 * @tparam U The scalar type.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param q The quantity.
 * @param scalar The scalar value.
 * @return A new quantity representing the quotient.
 */
template <typename T, arithmetic U, dimensional D, error_checking ErrorChecking>
constexpr auto operator/(const quantity<T, D, ErrorChecking> &q, const U &scalar) {
    using result_type = decltype(q.value() / scalar);

    if constexpr (ErrorChecking == error_checking::enabled) {
        quantity<result_type, D, ErrorChecking>::check_division_by_zero(scalar);
        quantity<result_type, D, ErrorChecking>::check_underflow_divide(q.value(), scalar);
    }

    return quantity<result_type, D, ErrorChecking>(q.value() / scalar);
}

/**
 * @brief Division operator for scalar and quantity.
 *
 * @tparam T The scalar type.
 * @tparam U The underlying type of the quantity.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param scalar The scalar value.
 * @param q The quantity.
 * @return A new quantity representing the quotient.
 */
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

/**
 * @brief Output stream operator for quantities.
 *
 * @tparam T The underlying type of the quantity.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param os The output stream.
 * @param q The quantity to output.
 * @return The output stream.
 */
template <arithmetic T, dimensional D, error_checking ErrorChecking>
auto operator<<(std::ostream &os, const quantity<T, D, ErrorChecking> &q) -> std::ostream & {
    return os << q.value();
}

/**
 * @brief Input stream operator for quantities.
 *
 * @tparam T The underlying type of the quantity.
 * @tparam D The dimension of the quantity.
 * @tparam ErrorChecking Error checking policy of the quantity.
 * @param is The input stream.
 * @param q The quantity to input into.
 * @return The input stream.
 */
template <arithmetic T, dimensional D, error_checking ErrorChecking>
auto operator>>(std::istream &is, quantity<T, D, ErrorChecking> &q) -> std::istream & {
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