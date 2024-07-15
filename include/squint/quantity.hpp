/**
 * @file quantity.hpp
 * @author Brendan Barnes
 * @brief Compile-time quantity types. Used to define units and constants
 *
 */

#ifndef SQUINT_QUANTITY_HPP
#define SQUINT_QUANTITY_HPP

#include "squint/core.hpp"
#include "squint/dimension.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <concepts>
#include <iostream>
#include <numbers>
#include <type_traits>

namespace squint {
template <error_checking ErrorChecking1, error_checking ErrorChecking2> struct resulting_error_checking {
    static constexpr auto value = ErrorChecking1 == error_checking::enabled || ErrorChecking2 == error_checking::enabled
                                      ? error_checking::enabled
                                      : error_checking::disabled;
};

namespace detail {
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
} // namespace detail

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
        return quantity<T, new_dimension>(detail::int_pow(value_, N));
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
        return quantity<T, new_dimension>(detail::sqrt_constexpr(value_));
    }

  private:
    T value_;
};

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

// Type alias for quantities with error checking enabled
template <typename T, dimensional D> using checked_quantity = quantity<T, D, error_checking::enabled>;

// Type alias specifically for constants (always uses error_checking_disabled)
template <typename T, dimensional D> using constant_quantity = quantity<T, D, error_checking::disabled>;

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

// approx equal for quantities
template <quantitative T, quantitative U, arithmetic Epsilon>
bool approx_equal(const T &a, const U &b, const Epsilon &epsilon = Epsilon{128 * 1.192092896e-04}) {
    static_assert(std::is_same_v<typename T::dimension_type, typename U::dimension_type>,
                  "Quantities must have the same dimension");
    return approx_equal(a.value(), b.value(), epsilon);
}

// approx equal for mixed types
template <quantitative T, arithmetic U, arithmetic V>
bool approx_equal(const T &a, const U &b, const V &epsilon = V{128 * 1.192092896e-04})
    requires std::is_same_v<typename T::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a.value(), b, epsilon);
}

template <arithmetic T, quantitative U, arithmetic V>
bool approx_equal(const T &a, const U &b, const V &epsilon = V{128 * 1.192092896e-04})
    requires std::is_same_v<typename U::dimension_type, dimensions::dimensionless>
{
    return approx_equal(a, b.value(), epsilon);
}

namespace units {

// Base unit type
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
struct unit_t : quantity<T, D, ErrorChecking> {
    using quantity<T, D, ErrorChecking>::quantity;
    static constexpr T convert_to(const unit_t &u, const unit_t & /*unused*/) { return u.value(); }
    // Allow implicit conversion from quantity<T, D, ErrorChecking>
    constexpr unit_t(const quantity<T, D, ErrorChecking> &q) : unit_t<T, D, ErrorChecking>(q.value()) {}
};

// Dimensionless
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dimensionless_t = unit_t<T, dimensions::dimensionless, ErrorChecking>;

// Length
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct length_t : unit_t<T, dimensions::length, ErrorChecking> {
    using unit_t<T, dimensions::length, ErrorChecking>::unit_t;
    static constexpr length_t<T, ErrorChecking> meters(T value) { return length_t<T, ErrorChecking>(value); }
    static constexpr length_t<T, ErrorChecking> feet(T value) { return length_t<T, ErrorChecking>(value * T(0.3048)); }
    static constexpr length_t<T, ErrorChecking> inches(T value) {
        return length_t<T, ErrorChecking>(value * T(0.0254));
    }
    static constexpr length_t<T, ErrorChecking> kilometers(T value) {
        return length_t<T, ErrorChecking>(value * T(1000.0));
    }
    static constexpr length_t<T, ErrorChecking> miles(T value) {
        return length_t<T, ErrorChecking>(value * T(1609.344));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct feet_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const feet_t & /*unused*/) {
        return l.value() / T(0.3048);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct inches_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const inches_t & /*unused*/) {
        return l.value() / T(0.0254);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilometers_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const kilometers_t & /*unused*/) {
        return l.value() / T(1000.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct miles_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const miles_t & /*unused*/) {
        return l.value() / T(1609.344);
    }
};

// Time
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct time_t : unit_t<T, dimensions::time, ErrorChecking> {
    using unit_t<T, dimensions::time, ErrorChecking>::unit_t;
    static constexpr time_t<T, ErrorChecking> seconds(T value) { return time_t<T, ErrorChecking>(value); }
    static constexpr time_t<T, ErrorChecking> minutes(T value) { return time_t<T, ErrorChecking>(value * T(60.0)); }
    static constexpr time_t<T, ErrorChecking> hours(T value) { return time_t<T, ErrorChecking>(value * T(3600.0)); }
    static constexpr time_t<T, ErrorChecking> days(T value) { return time_t<T, ErrorChecking>(value * T(86400.0)); }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct minutes_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const minutes_t & /*unused*/) {
        return t.value() / T(60.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct hours_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const hours_t & /*unused*/) {
        return t.value() / T(3600.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct days_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const days_t & /*unused*/) {
        return t.value() / T(86400.0);
    }
};

// Mass
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct mass_t : unit_t<T, dimensions::mass, ErrorChecking> {
    using unit_t<T, dimensions::mass, ErrorChecking>::unit_t;
    static constexpr mass_t<T, ErrorChecking> kilograms(T value) { return mass_t<T, ErrorChecking>(value); }
    static constexpr mass_t<T, ErrorChecking> grams(T value) { return mass_t<T, ErrorChecking>(value * T(0.001)); }
    static constexpr mass_t<T, ErrorChecking> pounds(T value) {
        return mass_t<T, ErrorChecking>(value * T(0.45359237));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct grams_t : mass_t<T, ErrorChecking> {
    using mass_t<T, ErrorChecking>::mass_t;
    static constexpr T convert_to(const mass_t<T, ErrorChecking> &m, const grams_t & /*unused*/) {
        return m.value() / T(0.001);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pounds_t : mass_t<T, ErrorChecking> {
    using mass_t<T, ErrorChecking>::mass_t;
    static constexpr T convert_to(const mass_t<T, ErrorChecking> &m, const pounds_t & /*unused*/) {
        return m.value() / T(0.45359237);
    }
};

// Temperature
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct temperature_t : unit_t<T, dimensions::temperature, ErrorChecking> {
    using unit_t<T, dimensions::temperature, ErrorChecking>::unit_t;
    static constexpr temperature_t<T, ErrorChecking> kelvin(T value) { return temperature_t<T, ErrorChecking>(value); }
    static constexpr temperature_t<T, ErrorChecking> celsius(T value) {
        return temperature_t<T, ErrorChecking>(value + T(273.15));
    }
    static constexpr temperature_t<T, ErrorChecking> fahrenheit(T value) {
        return temperature_t<T, ErrorChecking>((value - T(32.0)) * T(5.0) / T(9.0) + T(273.15));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct celsius_t : temperature_t<T, ErrorChecking> {
    using temperature_t<T, ErrorChecking>::temperature_t;
    static constexpr T convert_to(const temperature_t<T, ErrorChecking> &t, const celsius_t & /*unused*/) {
        return t.value() - T(273.15);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct fahrenheit_t : temperature_t<T, ErrorChecking> {
    using temperature_t<T, ErrorChecking>::temperature_t;
    static constexpr T convert_to(const temperature_t<T, ErrorChecking> &t, const fahrenheit_t & /*unused*/) {
        return t.value() * T(9.0) / T(5.0) - T(459.67);
    }
};

// Current
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using current_t = unit_t<T, dimensions::current, ErrorChecking>;

// Amount of substance
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using amount_of_substance_t = unit_t<T, dimensions::amount_of_substance, ErrorChecking>;

// Luminous intensity
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using luminous_intensity_t = unit_t<T, dimensions::luminous_intensity, ErrorChecking>;

// Angle
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct angle_t : unit_t<T, dimensions::dimensionless, ErrorChecking> {
    using unit_t<T, dimensions::dimensionless, ErrorChecking>::unit_t;
    static constexpr angle_t<T, ErrorChecking> radians(T value) { return angle_t<T, ErrorChecking>(value); }
    static constexpr angle_t<T, ErrorChecking> degrees(T value) {
        return angle_t<T, ErrorChecking>(value * std::numbers::pi_v<T> / T(180.0));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct degrees_t : angle_t<T, ErrorChecking> {
    using angle_t<T, ErrorChecking>::angle_t;
    static constexpr T convert_to(const angle_t<T, ErrorChecking> &a, const degrees_t & /*unused*/) {
        return a.value() * T(180.0) / std::numbers::pi_v<T>;
    }
};

// Velocity
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct velocity_t : unit_t<T, dimensions::velocity, ErrorChecking> {
    using unit_t<T, dimensions::velocity, ErrorChecking>::unit_t;
    static constexpr velocity_t<T, ErrorChecking> meters_per_second(T value) {
        return velocity_t<T, ErrorChecking>(value);
    }
    static constexpr velocity_t<T, ErrorChecking> kilometers_per_hour(T value) {
        return velocity_t<T, ErrorChecking>(value / T(3.6));
    }
    static constexpr velocity_t<T, ErrorChecking> miles_per_hour(T value) {
        return velocity_t<T, ErrorChecking>(value * T(0.44704));
    }
    static constexpr velocity_t<T, ErrorChecking> feet_per_second(T value) {
        return velocity_t<T, ErrorChecking>(value * T(3.28084));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilometers_per_hour_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const kilometers_per_hour_t & /*unused*/) {
        return v.value() * T(3.6);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct miles_per_hour_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const miles_per_hour_t & /*unused*/) {
        return v.value() / T(0.44704);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct feet_per_second_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const feet_per_second_t & /*unused*/) {
        return v.value() / T(3.28084);
    }
};

// Acceleration
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct acceleration_t : unit_t<T, dimensions::acceleration, ErrorChecking> {
    using unit_t<T, dimensions::acceleration, ErrorChecking>::unit_t;
    static constexpr acceleration_t<T, ErrorChecking> meters_per_second_squared(T value) {
        return acceleration_t<T, ErrorChecking>(value);
    }
};

// Area
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct area_t : unit_t<T, dimensions::area, ErrorChecking> {
    using unit_t<T, dimensions::area, ErrorChecking>::unit_t;
    static constexpr area_t<T, ErrorChecking> square_meters(T value) { return area_t<T, ErrorChecking>(value); }
    static constexpr area_t<T, ErrorChecking> square_feet(T value) {
        return area_t<T, ErrorChecking>(value * T(0.09290304));
    }
    static constexpr area_t<T, ErrorChecking> acres(T value) {
        return area_t<T, ErrorChecking>(value * T(4046.8564224));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct square_feet_t : area_t<T, ErrorChecking> {
    using area_t<T, ErrorChecking>::area_t;
    static constexpr T convert_to(const area_t<T, ErrorChecking> &a, const square_feet_t & /*unused*/) {
        return a.value() / T(0.09290304);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct acres_t : area_t<T, ErrorChecking> {
    using area_t<T, ErrorChecking>::area_t;
    static constexpr T convert_to(const area_t<T, ErrorChecking> &a, const acres_t & /*unused*/) {
        return a.value() / T(4046.8564224);
    }
};

// Volume
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct volume_t : unit_t<T, dimensions::volume, ErrorChecking> {
    using unit_t<T, dimensions::volume, ErrorChecking>::unit_t;
    static constexpr volume_t<T, ErrorChecking> cubic_meters(T value) { return volume_t<T, ErrorChecking>(value); }
    static constexpr volume_t<T, ErrorChecking> liters(T value) { return volume_t<T, ErrorChecking>(value * T(0.001)); }
    static constexpr volume_t<T, ErrorChecking> gallons(T value) {
        return volume_t<T, ErrorChecking>(value * T(0.00378541));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct liters_t : volume_t<T, ErrorChecking> {
    using volume_t<T, ErrorChecking>::volume_t;
    static constexpr T convert_to(const volume_t<T, ErrorChecking> &v, const liters_t & /*unused*/) {
        return v.value() / T(0.001);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct gallons_t : volume_t<T, ErrorChecking> {
    using volume_t<T, ErrorChecking>::volume_t;
    static constexpr T convert_to(const volume_t<T, ErrorChecking> &v, const gallons_t & /*unused*/) {
        return v.value() / T(0.00378541);
    }
};

// Force
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct force_t : unit_t<T, dimensions::force, ErrorChecking> {
    using unit_t<T, dimensions::force, ErrorChecking>::unit_t;
    static constexpr force_t<T, ErrorChecking> newtons(T value) { return force_t<T, ErrorChecking>(value); }
    static constexpr force_t<T, ErrorChecking> pounds_force(T value) {
        return force_t<T, ErrorChecking>(value * T(4.448222));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pounds_force_t : force_t<T, ErrorChecking> {
    using force_t<T, ErrorChecking>::force_t;
    static constexpr T convert_to(const force_t<T, ErrorChecking> &f, const pounds_force_t & /*unused*/) {
        return f.value() / T(4.448222);
    }
};

// Pressure
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pressure_t : unit_t<T, dimensions::pressure, ErrorChecking> {
    using unit_t<T, dimensions::pressure, ErrorChecking>::unit_t;
    static constexpr pressure_t<T, ErrorChecking> pascals(T value) { return pressure_t<T, ErrorChecking>(value); }
    static constexpr pressure_t<T, ErrorChecking> bars(T value) {
        return pressure_t<T, ErrorChecking>(value * T(100000.0));
    }
    static constexpr pressure_t<T, ErrorChecking> psi(T value) {
        return pressure_t<T, ErrorChecking>(value * T(6894.75729));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct bars_t : pressure_t<T, ErrorChecking> {
    using pressure_t<T, ErrorChecking>::pressure_t;
    static constexpr T convert_to(const pressure_t<T, ErrorChecking> &p, const bars_t & /*unused*/) {
        return p.value() / T(100000.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct psi_t : pressure_t<T, ErrorChecking> {
    using pressure_t<T, ErrorChecking>::pressure_t;
    static constexpr T convert_to(const pressure_t<T, ErrorChecking> &p, const psi_t & /*unused*/) {
        return p.value() / T(6894.75729);
    }
};

// Energy
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct energy_t : unit_t<T, dimensions::energy, ErrorChecking> {
    using unit_t<T, dimensions::energy, ErrorChecking>::unit_t;
    static constexpr energy_t<T, ErrorChecking> joules(T value) { return energy_t<T, ErrorChecking>(value); }
    static constexpr energy_t<T, ErrorChecking> kilowatt_hours(T value) {
        return energy_t<T, ErrorChecking>(value * T(3600000.0));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilowatt_hours_t : energy_t<T, ErrorChecking> {
    using energy_t<T, ErrorChecking>::energy_t;
    static constexpr T convert_to(const energy_t<T, ErrorChecking> &e, const kilowatt_hours_t & /*unused*/) {
        return e.value() / T(3600000.0);
    }
};

// Power
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct power_t : unit_t<T, dimensions::power, ErrorChecking> {
    using unit_t<T, dimensions::power, ErrorChecking>::unit_t;
    static constexpr power_t<T, ErrorChecking> watts(T value) { return power_t<T, ErrorChecking>(value); }
    static constexpr power_t<T, ErrorChecking> horsepower(T value) {
        return power_t<T, ErrorChecking>(value * T(745.699872));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct horsepower_t : power_t<T, ErrorChecking> {
    using power_t<T, ErrorChecking>::power_t;
    static constexpr T convert_to(const power_t<T, ErrorChecking> &p, const horsepower_t & /*unused*/) {
        return p.value() / T(745.699872);
    }
};

// Other derived units
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using density_t = unit_t<T, dimensions::density, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using charge_t = unit_t<T, dimensions::charge, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using voltage_t = unit_t<T, dimensions::voltage, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using capacitance_t = unit_t<T, dimensions::capacitance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using resistance_t = unit_t<T, dimensions::resistance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using conductance_t = unit_t<T, dimensions::conductance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using magnetic_flux_t = unit_t<T, dimensions::magnetic_flux, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using magnetic_flux_density_t = unit_t<T, dimensions::magnetic_flux_density, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using inductance_t = unit_t<T, dimensions::inductance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using frequency_t = unit_t<T, dimensions::frequency, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using angular_velocity_t = unit_t<T, dimensions::angular_velocity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using momentum_t = unit_t<T, dimensions::momentum, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using angular_momentum_t = unit_t<T, dimensions::angular_momentum, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using torque_t = unit_t<T, dimensions::torque, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using surface_tension_t = unit_t<T, dimensions::surface_tension, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dynamic_viscosity_t = unit_t<T, dimensions::dynamic_viscosity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using kinematic_viscosity_t = unit_t<T, dimensions::kinematic_viscosity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using heat_capacity_t = unit_t<T, dimensions::heat_capacity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using specific_heat_capacity_t = unit_t<T, dimensions::specific_heat_capacity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using thermal_conductivity_t = unit_t<T, dimensions::thermal_conductivity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using electric_field_strength_t = unit_t<T, dimensions::electric_field_strength, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using electric_displacement_t = unit_t<T, dimensions::electric_displacement, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using permittivity_t = unit_t<T, dimensions::permittivity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using permeability_t = unit_t<T, dimensions::permeability, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using molar_energy_t = unit_t<T, dimensions::molar_energy, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using molar_entropy_t = unit_t<T, dimensions::molar_entropy, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using exposure_t = unit_t<T, dimensions::exposure, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dose_equivalent_t = unit_t<T, dimensions::dose_equivalent, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using catalytic_activity_t = unit_t<T, dimensions::catalytic_activity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using wave_number_t = unit_t<T, dimensions::wave_number, ErrorChecking>;

// Convenience typedefs for float types with error checking disabled
using dimensionless = dimensionless_t<float>;
using length = length_t<float>;
using time = time_t<float>;
using mass = mass_t<float>;
using temperature = temperature_t<float>;
using current = current_t<float>;
using amount_of_substance = amount_of_substance_t<float>;
using luminous_intensity = luminous_intensity_t<float>;
using angle = angle_t<float>;
using velocity = velocity_t<float>;
using acceleration = acceleration_t<float>;
using area = area_t<float>;
using volume = volume_t<float>;
using force = force_t<float>;
using pressure = pressure_t<float>;
using energy = energy_t<float>;
using power = power_t<float>;
using density = density_t<float>;
using charge = charge_t<float>;
using voltage = voltage_t<float>;
using capacitance = capacitance_t<float>;
using resistance = resistance_t<float>;
using conductance = conductance_t<float>;
using magnetic_flux = magnetic_flux_t<float>;
using magnetic_flux_density = magnetic_flux_density_t<float>;
using inductance = inductance_t<float>;
using frequency = frequency_t<float>;
using angular_velocity = angular_velocity_t<float>;
using momentum = momentum_t<float>;
using angular_momentum = angular_momentum_t<float>;
using torque = torque_t<float>;
using surface_tension = surface_tension_t<float>;
using dynamic_viscosity = dynamic_viscosity_t<float>;
using kinematic_viscosity = kinematic_viscosity_t<float>;
using heat_capacity = heat_capacity_t<float>;
using specific_heat_capacity = specific_heat_capacity_t<float>;
using thermal_conductivity = thermal_conductivity_t<float>;
using electric_field_strength = electric_field_strength_t<float>;
using electric_displacement = electric_displacement_t<float>;
using permittivity = permittivity_t<float>;
using permeability = permeability_t<float>;
using molar_energy = molar_energy_t<float>;
using molar_entropy = molar_entropy_t<float>;
using exposure = exposure_t<float>;
using dose_equivalent = dose_equivalent_t<float>;
using catalytic_activity = catalytic_activity_t<float>;
using wave_number = wave_number_t<float>;

} // namespace units

namespace constants {

// Concept to ensure the type is a floating-point type
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Template alias for constant_quantity
template <FloatingPoint T, typename Dimension> using constant_quantity_t = constant_quantity<T, Dimension>;

// Mathematical constants
template <FloatingPoint T> struct math_constants {
    static constexpr auto pi = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::pi_v<T>);
    static constexpr auto e = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::e_v<T>);
    static constexpr auto sqrt2 = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::sqrt2_v<T>);
    static constexpr auto ln2 = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::ln2_v<T>);
    static constexpr auto phi = constant_quantity_t<T, dimensions::dimensionless>(T(1.618033988749895));
};

// Physical constants (SI)
template <FloatingPoint T> struct si_constants {
    // Speed of light in vacuum
    static constexpr auto c = constant_quantity_t<T, dimensions::velocity>(T(299'792'458.0));

    // Planck constant
    static constexpr auto h = constant_quantity_t<T, mult_t<dimensions::energy, dimensions::time>>(T(6.62607015e-34));

    // Reduced Planck constant (h-bar)
    static constexpr auto hbar = h / (T(2) * math_constants<T>::pi);

    // Gravitational constant
    static constexpr auto G =
        constant_quantity_t<T, div_t<mult_t<dimensions::force, dimensions::area>, pow_t<dimensions::mass, 2>>>(
            T(6.67430e-11));

    // Elementary charge
    static constexpr auto e_charge = constant_quantity_t<T, dimensions::charge>(T(1.602176634e-19));

    // Electron mass
    static constexpr auto m_e = constant_quantity_t<T, dimensions::mass>(T(9.1093837015e-31));

    // Proton mass
    static constexpr auto m_p = constant_quantity_t<T, dimensions::mass>(T(1.67262192369e-27));

    // Fine-structure constant
    static constexpr auto alpha = constant_quantity_t<T, dimensions::dimensionless>(T(7.2973525693e-3));

    // Boltzmann constant
    static constexpr auto k_B =
        constant_quantity_t<T, div_t<dimensions::energy, dimensions::temperature>>(T(1.380649e-23));

    // Avogadro constant
    static constexpr auto N_A = constant_quantity_t<T, inv_t<dimensions::amount_of_substance>>(T(6.02214076e23));

    // Gas constant
    static constexpr auto R = k_B * N_A;

    // Vacuum electric permittivity
    static constexpr auto epsilon_0 =
        constant_quantity_t<T, div_t<dimensions::capacitance, dimensions::length>>(T(8.8541878128e-12));

    // Vacuum magnetic permeability
    static constexpr auto mu_0 =
        constant_quantity_t<T, div_t<dimensions::inductance, dimensions::length>>(T(1.25663706212e-6));

    // Stefan-Boltzmann constant
    static constexpr auto sigma =
        constant_quantity_t<T, div_t<dimensions::power, mult_t<dimensions::area, pow_t<dimensions::temperature, 4>>>>(
            T(5.670374419e-8));
};

// Astronomical constants
template <FloatingPoint T> struct astro_constants {
    // Astronomical Unit
    static constexpr auto AU = constant_quantity_t<T, dimensions::length>(T(1.495978707e11));

    // Parsec
    static constexpr auto parsec = constant_quantity_t<T, dimensions::length>(T(3.0856775814913673e16));

    // Light year
    static constexpr auto light_year =
        si_constants<T>::c * constant_quantity_t<T, dimensions::time>(T(365.25) * T(24) * T(3600));

    // Solar mass
    static constexpr auto solar_mass = constant_quantity_t<T, dimensions::mass>(T(1.988847e30));

    // Earth mass
    static constexpr auto earth_mass = constant_quantity_t<T, dimensions::mass>(T(5.97217e24));

    // Earth radius (equatorial)
    static constexpr auto earth_radius = constant_quantity_t<T, dimensions::length>(T(6.3781e6));

    // Standard gravitational acceleration on Earth
    static constexpr auto g = constant_quantity_t<T, dimensions::acceleration>(T(9.80665));
};

// Atomic and nuclear constants
template <FloatingPoint T> struct atomic_constants {
    // Rydberg constant
    static constexpr auto R_inf = constant_quantity_t<T, dimensions::wave_number>(T(10973731.568160));

    // Bohr radius
    static constexpr auto a_0 = constant_quantity_t<T, dimensions::length>(T(5.29177210903e-11));

    // Classical electron radius
    static constexpr auto r_e = constant_quantity_t<T, dimensions::length>(T(2.8179403262e-15));

    // Proton-electron mass ratio
    static constexpr auto m_p_m_e = si_constants<T>::m_p / si_constants<T>::m_e;
};

} // namespace constants

namespace math {

// Absolute value
template <typename T> auto abs(const T &x) {
    if constexpr (quantitative<T>) {
        return T(std::abs(static_cast<typename T::value_type>(x)));
    } else {
        return std::abs(x);
    }
}

// Square root
template <typename T> auto sqrt(const T &x) {
    if constexpr (quantitative<T>) {
        using result_dimension = root_t<typename T::dimension_type, 2>;
        return quantity<decltype(std::sqrt(static_cast<typename T::value_type>(x))), result_dimension>(
            std::sqrt(static_cast<typename T::value_type>(x)));
    } else {
        return std::sqrt(x);
    }
}

// Exponential function (only for dimensionless quantities)
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

// Natural logarithm (only for dimensionless quantities)
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

// Power function
template <int N> auto pow(quantitative auto x) {
    using result_dimension = pow_t<typename decltype(x)::dimension_type, N>;
    return quantity<decltype(std::pow(static_cast<typename decltype(x)::value_type>(x), N)), result_dimension>(
        std::pow(static_cast<typename decltype(x)::value_type>(x), N));
}

// Trigonometric functions (only for dimensionless quantities)
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

// Inverse trigonometric functions (return dimensionless quantities)
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

// Two-argument arctangent
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

// Hyperbolic functions (only for dimensionless quantities)
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

// Inverse hyperbolic functions (return dimensionless quantities)
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

} // namespace math

template <typename T>
concept dimensionless_quantity =
    quantitative<T> && std::is_same_v<typename T::dimension_type, dimensions::dimensionless>;

} // namespace squint

#endif // SQUINT_QUANTITY_HPP