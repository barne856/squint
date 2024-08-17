/**
 * @file quantity.hpp
 * @brief Defines the quantity class for representing physical quantities with dimensions.
 */

#ifndef SQUINT_QUANTITY_QUANTITY_HPP
#define SQUINT_QUANTITY_QUANTITY_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/quantity/dimension_types.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace squint {

/**
 * @brief Represents a physical quantity with a value and dimension.
 *
 * @tparam T The arithmetic type used to represent the value.
 * @tparam D The dimension type representing the physical dimension.
 * @tparam E The error checking policy.
 */
template <arithmetic T, dimensional D, error_checking E = error_checking::disabled> class quantity {
  public:
    using value_type = T;
    using dimension_type = D;

    /// @name Constructors and Assignment
    /// @{

    /// @brief Default constructor
    constexpr quantity() noexcept : value_(T{}) {}

    /// @brief Destructor
    ~quantity() = default;

    /// @brief Explicit constructor from value
    constexpr explicit quantity(const T &value) noexcept : value_(value) {}

    /// @brief Copy constructor
    constexpr quantity(const quantity &) noexcept = default;

    /// @brief Move constructor
    constexpr quantity(quantity &&) noexcept = default;

    /// @brief Copy assignment operator
    constexpr auto operator=(const quantity &) noexcept -> quantity & = default;

    /// @brief Move assignment operator
    constexpr auto operator=(quantity &&) noexcept -> quantity & = default;

    /// @}

    /// @name Conversion Constructors and Operators
    /// @{

    /**
     * @brief Implicit conversion constructor for dimensionless quantities from arithmetic types
     * @tparam U Arithmetic type to convert from
     */
    template <arithmetic U>
    constexpr quantity(const U &value) noexcept
        requires std::is_same_v<D, dimensions::unity>
        : value_(static_cast<T>(value)) {}

    /**
     * @brief Explicit conversion operator to other types
     * @tparam U Type to convert to
     */
    template <typename U> explicit operator U() const noexcept { return U(value_); }

    /**
     * @brief Explicit conversion operator for non-dimensionless quantities
     */
    explicit constexpr operator T() const noexcept
        requires(!std::is_same_v<D, dimensions::unity>)
    {
        return value_;
    }

    /**
     * @brief Implicit conversion operator for dimensionless quantities
     */
    constexpr operator T() const noexcept
        requires std::is_same_v<D, dimensions::unity>
    {
        return value_;
    }

    /**
     * @brief Conversion constructor for quantities with different error checking
     * @tparam U Value type of the other quantity
     * @tparam OtherErrorChecking Error checking policy of the other quantity
     */
    template <typename U, error_checking OtherErrorChecking>
    constexpr quantity(const quantity<U, D, OtherErrorChecking> &other) noexcept
        : value_(static_cast<T>(other.value())) {
        if constexpr (E == error_checking::enabled && OtherErrorChecking == error_checking::disabled) {
            // Perform any necessary error checking here
        }
    }

    /// @}

    /// @name Accessors
    /// @{

    [[nodiscard]] constexpr auto value() noexcept -> T & { return value_; }
    [[nodiscard]] constexpr auto value() const noexcept -> const T & { return value_; }

    /// @}

    /// @name Error Checking Methods
    /// @{

    /**
     * @brief Get the error checking policy
     */
    [[nodiscard]] static constexpr auto error_checking() noexcept -> error_checking { return E; }

    /**
     * @brief Check for multiplication overflow
     * @tparam U Type of the second operand
     * @param a First operand
     * @param b Second operand
     */
    template <typename U> static constexpr void check_overflow_multiply(const T &a, const U &b) {
        if constexpr (E == error_checking::enabled && std::is_integral_v<T> && std::is_integral_v<U>) {
            if ((a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) ||
                (a < 0 && b < 0 && a < std::numeric_limits<T>::max() / b) ||
                (a > 0 && b < 0 && b < std::numeric_limits<T>::min() / a) ||
                (a < 0 && b > 0 && a < std::numeric_limits<T>::min() / b)) {
                throw std::overflow_error("Multiplication would cause overflow");
            }
        }
    }

    /**
     * @brief Check for division by zero
     * @tparam U Type of the divisor
     * @param b Divisor
     */
    template <typename U> static constexpr void check_division_by_zero(const U &b) {
        if constexpr (E == error_checking::enabled) {
            if (b == U(0)) {
                throw std::domain_error("Division by zero");
            }
        }
    }

    /**
     * @brief Check for division underflow
     * @tparam U Type of the divisor
     * @param a Dividend
     * @param b Divisor
     */
    template <typename U> static constexpr void check_underflow_divide(const T &a, const U &b) {
        if constexpr (E == error_checking::enabled && std::is_floating_point_v<T>) {
            if (std::abs(a) < std::numeric_limits<T>::min() * std::abs(b)) {
                throw std::underflow_error("Division would cause underflow");
            }
        }
    }

    /**
     * @brief Check for addition overflow
     * @param a First operand
     * @param b Second operand
     */
    static constexpr void check_overflow_add(const T &a, const T &b) {
        if constexpr (E == error_checking::enabled && std::is_integral_v<T>) {
            if ((b > 0 && a > std::numeric_limits<T>::max() - b) || (b < 0 && a < std::numeric_limits<T>::min() - b)) {
                throw std::overflow_error("Addition would cause overflow");
            }
        }
    }

    /**
     * @brief Check for subtraction underflow
     * @param a First operand
     * @param b Second operand
     */
    static constexpr void check_underflow_subtract(const T &a, const T &b) {
        if constexpr (E == error_checking::enabled && std::is_integral_v<T>) {
            if ((b < 0 && a > std::numeric_limits<T>::max() + b) || (b > 0 && a < std::numeric_limits<T>::min() + b)) {
                throw std::underflow_error("Subtraction would cause underflow");
            }
        }
    }

    /// @}

    /// @name Arithmetic Operators
    /// @{

    /**
     * @brief Multiply-assign operator
     * @tparam U Type of the scalar
     * @param scalar Scalar to multiply by
     * @return Reference to this quantity
     */
    template <typename U>
    constexpr auto operator*=(const U &scalar) -> quantity &
        requires(arithmetic<U> || std::is_same_v<typename U::dimension_type, dimensions::unity>)
    {
        check_overflow_multiply(value_, scalar);
        value_ *= scalar;
        return *this;
    }

    /**
     * @brief Divide-assign operator
     * @tparam U Type of the scalar
     * @param scalar Scalar to divide by
     * @return Reference to this quantity
     */
    template <typename U>
    constexpr auto operator/=(const U &scalar) -> quantity &
        requires(arithmetic<U> || std::is_same_v<typename U::dimension_type, dimensions::unity>)
    {
        check_division_by_zero(scalar);
        check_underflow_divide(value_, scalar);
        value_ /= scalar;
        return *this;
    }

    /**
     * @brief Add-assign operator
     * @param rhs Quantity to add
     * @return Reference to this quantity
     */
    constexpr auto operator+=(const quantity &rhs) -> quantity & {
        check_overflow_add(value_, rhs.value_);
        value_ += rhs.value_;
        return *this;
    }

    /**
     * @brief Subtract-assign operator
     * @param rhs Quantity to subtract
     * @return Reference to this quantity
     */
    constexpr auto operator-=(const quantity &rhs) -> quantity & {
        check_underflow_subtract(value_, rhs.value_);
        value_ -= rhs.value_;
        return *this;
    }

    /**
     * @brief Unary negation operator
     * @return Negated quantity
     */
    constexpr auto operator-() const noexcept -> quantity { return quantity(-value_); }

    /**
     * @brief Pre-increment operator
     * @return Reference to this quantity
     */
    constexpr auto operator++() noexcept -> quantity & {
        ++value_;
        return *this;
    }

    /**
     * @brief Post-increment operator
     * @return Copy of the quantity before increment
     */
    constexpr auto operator++(int) noexcept -> quantity {
        quantity temp(*this);
        ++(*this);
        return temp;
    }

    /**
     * @brief Pre-decrement operator
     * @return Reference to this quantity
     */
    constexpr auto operator--() noexcept -> quantity & {
        --value_;
        return *this;
    }

    /**
     * @brief Post-decrement operator
     * @return Copy of the quantity before decrement
     */
    constexpr auto operator--(int) noexcept -> quantity {
        quantity temp(*this);
        --(*this);
        return temp;
    }

    /// @}

    /// @name Comparison Operators
    /// @{

    /**
     * @brief Three-way comparison operator
     * @param rhs Quantity to compare with
     * @return Comparison result
     */
    constexpr auto operator<=>(const quantity &rhs) const noexcept { return value_ <=> rhs.value_; }

    /**
     * @brief Equality comparison operator
     * @param rhs Quantity to compare with
     * @return True if quantities are equal, false otherwise
     */
    constexpr auto operator==(const quantity &rhs) const noexcept -> bool { return value_ == rhs.value_; }

    /// @}

  private:
    T value_;
};

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_HPP