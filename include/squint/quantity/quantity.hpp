#ifndef SQUINT_QUANTITY_QUANTITY_HPP
#define SQUINT_QUANTITY_QUANTITY_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/quantity/dimension.hpp"
#include "squint/util/math_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <compare>
#include <concepts>
#include <iostream>
#include <limits>
#include <numbers>
#include <type_traits>

namespace squint {

template <arithmetic T, dimensional D, error_checking E = error_checking::disabled> class quantity {
  public:
    using value_type = T;
    using dimension_type = D;

    // Constructors
    constexpr quantity() noexcept : value_(T{}) {}
    constexpr explicit quantity(const T &value) noexcept : value_(value) {}
    explicit quantity(const quantity &) noexcept = default;
    explicit quantity(quantity &&) noexcept = default;

    // Assignment operators
    constexpr quantity &operator=(const quantity &) noexcept = default;
    constexpr quantity &operator=(quantity &&) noexcept = default;

    // Conditional implicit conversion constructor (for dimensionless quantities from arithmetic types)
    template <arithmetic U>
    constexpr quantity(const U &value) noexcept
        requires std::is_same_v<D, dimensions::dimensionless>
        : value_(static_cast<T>(value)) {}

    // Explicit conversion constructor for all other cases (you can static_cast from arithmetic types if you want to)
    template <arithmetic U>
    explicit constexpr quantity(const U &value) noexcept
        requires(!std::is_same_v<D, dimensions::dimensionless>)
        : value_(static_cast<T>(value)) {}

    // Allow explicit cast to other types
    template <typename U> explicit operator U() const noexcept { return U(value_); }

    // Explicit conversion operator for non-dimensionless quantities
    explicit constexpr operator T() const noexcept
        requires(!std::is_same_v<D, dimensions::dimensionless>)
    {
        return value_;
    }

    // Implicit conversion operator for dimensionless quantities
    constexpr operator T() const noexcept
        requires std::is_same_v<D, dimensions::dimensionless>
    {
        return value_;
    }

    // Conversion constructor for quantities with different error checking
    template <typename U, error_checking OtherErrorChecking>
    constexpr quantity(const quantity<U, D, OtherErrorChecking> &other) noexcept
        : value_(static_cast<T>(other.value())) {
        if constexpr (E == error_checking::enabled && OtherErrorChecking == error_checking::disabled) {
            // Perform any necessary error checking here
        }
    }

    // Accessor methods
    [[nodiscard]] constexpr T &value() noexcept { return value_; }
    [[nodiscard]] constexpr const T &value() const noexcept { return value_; }
    [[nodiscard]] constexpr const T *operator->() const noexcept { return &value_; }
    [[nodiscard]] constexpr T *operator->() noexcept { return &value_; }
    [[nodiscard]] constexpr const T &operator*() const noexcept { return value_; }
    [[nodiscard]] constexpr T &operator*() noexcept { return value_; }

    // Error checking methods
    template <typename U> static constexpr void check_overflow_multiply(const T &a, const U &b) {
        if constexpr (E == error_checking::enabled) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
                if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
                    throw std::overflow_error("Multiplication would cause overflow");
                }
                if (a < 0 && b < 0 && a < std::numeric_limits<T>::max() / b) {
                    throw std::overflow_error("Multiplication would cause overflow");
                }
                if (a > 0 && b < 0 && b < std::numeric_limits<T>::min() / a) {
                    throw std::overflow_error("Multiplication would cause overflow");
                }
                if (a < 0 && b > 0 && a < std::numeric_limits<T>::min() / b) {
                    throw std::overflow_error("Multiplication would cause overflow");
                }
            }
        }
    }

    template <typename U> static constexpr void check_division_by_zero(const U &b) {
        if constexpr (E == error_checking::enabled) {
            if (b == U(0)) {
                throw std::domain_error("Division by zero");
            }
        }
    }

    template <typename U> static constexpr void check_underflow_divide(const T &a, const U &b) {
        if constexpr (E == error_checking::enabled) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::abs(a) < std::numeric_limits<T>::min() * std::abs(b)) {
                    throw std::underflow_error("Division would cause underflow");
                }
            }
        }
    }

    static constexpr void check_overflow_add(const T &a, const T &b) {
        if constexpr (E == error_checking::enabled) {
            if constexpr (std::is_integral_v<T>) {
                if (b > 0 && a > std::numeric_limits<T>::max() - b) {
                    throw std::overflow_error("Addition would cause overflow");
                }
                if (b < 0 && a < std::numeric_limits<T>::min() - b) {
                    throw std::overflow_error("Addition would cause overflow");
                }
            }
        }
    }

    static constexpr void check_overflow_subtract(const T &a, const T &b) {
        if constexpr (E == error_checking::enabled) {
            if constexpr (std::is_integral_v<T>) {
                if (b < 0 && a > std::numeric_limits<T>::max() + b) {
                    throw std::overflow_error("Subtraction would cause overflow");
                }
                if (b > 0 && a < std::numeric_limits<T>::min() + b) {
                    throw std::overflow_error("Subtraction would cause overflow");
                }
            }
        }
    }

    // Arithmetic operators
    template <typename U>
    constexpr quantity &operator*=(const U &scalar)
        requires(arithmetic<U> || std::is_same_v<typename U::dimension_type, dimensions::dimensionless>)
    {
        check_overflow_multiply(value_, scalar);
        value_ *= scalar;
        return *this;
    }

    template <typename U>
    constexpr quantity &operator/=(const U &scalar)
        requires(arithmetic<U> || std::is_same_v<typename U::dimension_type, dimensions::dimensionless>)
    {
        check_division_by_zero(scalar);
        check_underflow_divide(value_, scalar);
        value_ /= scalar;
        return *this;
    }

    constexpr quantity &operator+=(const quantity &rhs) {
        check_overflow_add(value_, rhs.value_);
        value_ += rhs.value_;
        return *this;
    }

    constexpr quantity &operator-=(const quantity &rhs) {
        check_overflow_subtract(value_, rhs.value_);
        value_ -= rhs.value_;
        return *this;
    }

    // Unary negation operator
    constexpr quantity operator-() const noexcept { return quantity(-value_); }

    // Increment and decrement operators
    constexpr quantity &operator++() noexcept {
        ++value_;
        return *this;
    }

    constexpr quantity operator++(int) noexcept {
        quantity temp(*this);
        ++(*this);
        return temp;
    }

    constexpr quantity &operator--() noexcept {
        --value_;
        return *this;
    }

    constexpr quantity operator--(int) noexcept {
        quantity temp(*this);
        --(*this);
        return temp;
    }

    // Three-way comparison operator
    constexpr auto operator<=>(const quantity &rhs) const noexcept { return value_ <=> rhs.value_; }

    // Equality comparison
    constexpr bool operator==(const quantity &rhs) const noexcept { return value_ == rhs.value_; }

    // Unit conversion
    template <template <typename, error_checking> typename TargetUnit, error_checking TargetErrorChecking = E>
    constexpr auto as() const {
        if constexpr (std::is_same_v<TargetUnit<T, TargetErrorChecking>, quantity<T, D, E>>) {
            return value_;
        } else {
            return TargetUnit<T, TargetErrorChecking>::convert_to(*this, TargetUnit<T, TargetErrorChecking>{});
        }
    }

    // Power method
    template <int N> constexpr auto pow() const {
        using new_dimension = pow_t<D, N>;
        return quantity<T, new_dimension>(int_pow(value_, N));
    }

    // Root method
    template <int N> auto root() const {
        static_assert(N > 0, "Cannot take 0th root");
        using new_dimension = root_t<D, N>;
        // Note: This is not constexpr, as there's no general constexpr nth root
        return quantity<T, new_dimension>(std::pow(value_, T(1) / N));
    }

    // Square root method
    constexpr auto sqrt() const {
        using new_dimension = root_t<D, 2>;
        return quantity<T, new_dimension>(sqrt_constexpr(value_));
    }

  private:
    T value_;
};

template <typename T>
concept dimensionless_quantity =
    quantitative<T> && std::is_same_v<typename T::dimension_type, dimensions::dimensionless>;

template <typename T>
concept dimensionless_scalar = arithmetic<T> || dimensionless_quantity<T>;

// Type alias for quantities with error checking enabled
template <typename T, dimensional D> using checked_quantity = quantity<T, D, error_checking::enabled>;

// Type alias specifically for constants (always uses error_checking_disabled)
template <typename T, dimensional D> using constant_quantity = quantity<T, D, error_checking::disabled>;

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_HPP