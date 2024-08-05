/**
 * @file quantity_ops.hpp
 * @brief Defines arithmetic and stream operations for quantitative types.
 *
 * This file provides operator overloads for arithmetic operations between
 * quantitative types (such as quantities and units), between quantitative types
 * and scalars, and for stream I/O operations. It uses concepts to ensure type
 * safety and proper dimension handling in calculations.
 *
 * @note All operations return quantity types, even when operating on unit types,
 * to ensure consistent behavior and dimension tracking.
 *
 */

#ifndef SQUINT_QUANTITY_QUANTITY_OPS_HPP
#define SQUINT_QUANTITY_QUANTITY_OPS_HPP

#include "squint/core/concepts.hpp"
#include "squint/quantity/quantity.hpp"
#include <iostream>

namespace squint {

/**
 * @brief Addition operator for quantitative types.
 *
 * @tparam T1 The type of the left-hand operand.
 * @tparam T2 The type of the right-hand operand.
 * @param lhs The left-hand side quantitative value.
 * @param rhs The right-hand side quantitative value.
 * @return A new quantity representing the sum.
 *
 * @throws std::overflow_error if error checking is enabled and addition would cause overflow.
 *
 * @note Both operands must have the same dimension.
 */
template <quantitative T1, quantitative T2>
    requires std::is_same_v<typename T1::dimension_type, typename T2::dimension_type>
constexpr auto operator+(const T1 &lhs, const T2 &rhs) {
    using result_type = decltype(lhs.value() + rhs.value());
    using result_dimension = typename T1::dimension_type;
    constexpr error_checking result_error_checking =
        resulting_error_checking<T1::error_checking(), T2::error_checking()>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_overflow_add(lhs.value(), rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() + rhs.value());
}

/**
 * @brief Subtraction operator for quantitative types.
 *
 * @tparam T1 The type of the left-hand operand.
 * @tparam T2 The type of the right-hand operand.
 * @param lhs The left-hand side quantitative value.
 * @param rhs The right-hand side quantitative value.
 * @return A new quantity representing the difference.
 *
 * @throws std::underflow_error if error checking is enabled and subtraction would cause underflow.
 *
 * @note Both operands must have the same dimension.
 */
template <quantitative T1, quantitative T2>
    requires std::is_same_v<typename T1::dimension_type, typename T2::dimension_type>
constexpr auto operator-(const T1 &lhs, const T2 &rhs) {
    using result_type = decltype(lhs.value() - rhs.value());
    using result_dimension = typename T1::dimension_type;
    constexpr error_checking result_error_checking =
        resulting_error_checking<T1::error_checking(), T2::error_checking()>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_underflow_subtract(lhs.value(),
                                                                                                 rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() - rhs.value());
}

/**
 * @brief Multiplication operator for quantitative types.
 *
 * @tparam T1 The type of the left-hand operand.
 * @tparam T2 The type of the right-hand operand.
 * @param lhs The left-hand side quantitative value.
 * @param rhs The right-hand side quantitative value.
 * @return A new quantity representing the product.
 *
 * @throws std::overflow_error if error checking is enabled and multiplication would cause overflow.
 *
 * @note The resulting dimension is the product of the operands' dimensions.
 */
template <quantitative T1, quantitative T2> constexpr auto operator*(const T1 &lhs, const T2 &rhs) {
    using result_type = decltype(lhs.value() * rhs.value());
    using result_dimension = dim_mult_t<typename T1::dimension_type, typename T2::dimension_type>;
    constexpr error_checking result_error_checking =
        resulting_error_checking<T1::error_checking(), T2::error_checking()>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_overflow_multiply(lhs.value(),
                                                                                                rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() * rhs.value());
}

/**
 * @brief Division operator for quantitative types.
 *
 * @tparam T1 The type of the left-hand operand.
 * @tparam T2 The type of the right-hand operand.
 * @param lhs The left-hand side quantitative value.
 * @param rhs The right-hand side quantitative value.
 * @return A new quantity representing the quotient.
 *
 * @throws std::domain_error if error checking is enabled and the divisor is zero.
 * @throws std::underflow_error if error checking is enabled and division would cause underflow.
 *
 * @note The resulting dimension is the quotient of the operands' dimensions.
 */
template <quantitative T1, quantitative T2> constexpr auto operator/(const T1 &lhs, const T2 &rhs) {
    using result_type = decltype(lhs.value() / rhs.value());
    using result_dimension = dim_div_t<typename T1::dimension_type, typename T2::dimension_type>;
    constexpr error_checking result_error_checking =
        resulting_error_checking<T1::error_checking(), T2::error_checking()>::value;

    if constexpr (result_error_checking == error_checking::enabled) {
        quantity<result_type, result_dimension, result_error_checking>::check_division_by_zero(rhs.value());
        quantity<result_type, result_dimension, result_error_checking>::check_underflow_divide(lhs.value(),
                                                                                               rhs.value());
    }

    return quantity<result_type, result_dimension, result_error_checking>(lhs.value() / rhs.value());
}

/**
 * @brief Multiplication operator for scalar and quantitative type.
 *
 * @tparam T The scalar type.
 * @tparam U The quantitative type.
 * @param scalar The scalar value.
 * @param q The quantitative value.
 * @return A new quantity representing the product.
 *
 * @throws std::overflow_error if error checking is enabled and multiplication would cause overflow.
 */
template <arithmetic T, quantitative U> constexpr auto operator*(const T &scalar, const U &q) {
    using result_type = decltype(scalar * q.value());
    using result_dimension = typename U::dimension_type;

    if constexpr (U::error_checking() == error_checking::enabled) {
        quantity<result_type, result_dimension, U::error_checking()>::check_overflow_multiply(scalar, q.value());
    }

    return quantity<result_type, result_dimension, U::error_checking()>(scalar * q.value());
}

/**
 * @brief Multiplication operator for quantitative type and scalar.
 *
 * @tparam T The quantitative type.
 * @tparam U The scalar type.
 * @param q The quantitative value.
 * @param scalar The scalar value.
 * @return A new quantity representing the product.
 *
 * @note This operator delegates to the scalar-first multiplication operator.
 */
template <quantitative T, arithmetic U> constexpr auto operator*(const T &q, const U &scalar) { return scalar * q; }

/**
 * @brief Division operator for quantitative type and scalar.
 *
 * @tparam T The quantitative type.
 * @tparam U The scalar type.
 * @param q The quantitative value.
 * @param scalar The scalar value.
 * @return A new quantity representing the quotient.
 *
 * @throws std::domain_error if error checking is enabled and the scalar is zero.
 * @throws std::underflow_error if error checking is enabled and division would cause underflow.
 */
template <quantitative T, arithmetic U> constexpr auto operator/(const T &q, const U &scalar) {
    using result_type = decltype(q.value() / scalar);
    using result_dimension = typename T::dimension_type;

    if constexpr (T::error_checking() == error_checking::enabled) {
        quantity<result_type, result_dimension, T::error_checking()>::check_division_by_zero(scalar);
        quantity<result_type, result_dimension, T::error_checking()>::check_underflow_divide(q.value(), scalar);
    }

    return quantity<result_type, result_dimension, T::error_checking()>(q.value() / scalar);
}

/**
 * @brief Division operator for scalar and quantitative type.
 *
 * @tparam T The scalar type.
 * @tparam U The quantitative type.
 * @param scalar The scalar value.
 * @param q The quantitative value.
 * @return A new quantity representing the quotient.
 *
 * @throws std::domain_error if error checking is enabled and the quantitative value is zero.
 * @throws std::underflow_error if error checking is enabled and division would cause underflow.
 *
 * @note The resulting dimension is the inverse of the quantitative type's dimension.
 */
template <arithmetic T, quantitative U> constexpr auto operator/(const T &scalar, const U &q) {
    using result_type = decltype(scalar / q.value());
    using result_dimension = dim_inv_t<typename U::dimension_type>;

    if constexpr (U::error_checking() == error_checking::enabled) {
        quantity<result_type, result_dimension, U::error_checking()>::check_division_by_zero(q.value());
        quantity<result_type, result_dimension, U::error_checking()>::check_underflow_divide(scalar, q.value());
    }

    return quantity<result_type, result_dimension, U::error_checking()>(scalar / q.value());
}

/**
 * @brief Output stream operator for quantitative types.
 *
 * @tparam T The quantitative type.
 * @param os The output stream.
 * @param q The quantitative value to output.
 * @return The output stream.
 */
template <quantitative T> auto operator<<(std::ostream &os, const T &q) -> std::ostream & { return os << q.value(); }

/**
 * @brief Input stream operator for quantitative types.
 *
 * @tparam T The quantitative type.
 * @param is The input stream.
 * @param q The quantitative value to input into.
 * @return The input stream.
 *
 * @throws Any exception thrown by the quantitative type's constructor if error checking is enabled.
 */
template <quantitative T> auto operator>>(std::istream &is, T &q) -> std::istream & {
    typename T::value_type value;
    is >> value;
    if constexpr (T::error_checking() == error_checking::enabled) {
        try {
            q = T(value);
        } catch (const std::exception &e) {
            is.setstate(std::ios_base::failbit);
            throw;
        }
    } else {
        q = T(value);
    }
    return is;
}

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_OPS_HPP