#ifndef SQUINT_QUANTITY_UNIT_HPP
#define SQUINT_QUANTITY_UNIT_HPP

#include "squint/quantity/quantity.hpp"
#include <numbers>

namespace squint::units {

/// @brief Conversion constants for unit conversions
/// @{
inline constexpr auto FEET_TO_METERS = 0.3048;
inline constexpr auto INCHES_TO_METERS = 0.0254;
inline constexpr auto KILOMETERS_TO_METERS = 1000.0;
inline constexpr auto MILES_TO_METERS = 1609.344;
inline constexpr auto MINUTES_TO_SECONDS = 60.0;
inline constexpr auto HOURS_TO_SECONDS = 3600.0;
inline constexpr auto DAYS_TO_SECONDS = 86400.0;
inline constexpr auto GRAMS_TO_KILOGRAMS = 0.001;
inline constexpr auto POUNDS_TO_KILOGRAMS = 0.45359237;
inline constexpr auto CELSIUS_OFFSET = 273.15;
inline constexpr auto FAHRENHEIT_SCALE = 5.0 / 9.0;
inline constexpr auto FAHRENHEIT_OFFSET = 459.67;
inline constexpr auto KMH_TO_MS = 1.0 / 3.6;
inline constexpr auto MPH_TO_MS = 0.44704;
inline constexpr auto DEGREES_IN_HALF_CIRCLE = 180.0;
/// @}

/**
 * @brief Base unit type for all physical quantities.
 *
 * @tparam T The underlying numeric type (e.g., float, double).
 * @tparam D The dimension of the unit.
 * @tparam ErrorChecking The error checking policy.
 */
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
struct unit_t : quantity<T, D, ErrorChecking> {
    using quantity<T, D, ErrorChecking>::quantity;

    /**
     * @brief Convert from this unit to itself (identity conversion).
     *
     * @param u The unit to convert.
     * @return The value of the unit.
     */
    static constexpr auto convert_to(const unit_t &u, const unit_t & /*unused*/) -> T { return u.value(); }

    /**
     * @brief Construct a unit from a quantity of the same type.
     *
     * @param q The quantity to construct from.
     */
    constexpr unit_t(const quantity<T, D, ErrorChecking> &q) : unit_t<T, D, ErrorChecking>(q.value()) {}
};

/**
 * @brief Generalized helper struct for unit conversion using linear scaling.
 *
 * @tparam T The underlying numeric type.
 * @tparam D The dimension of the unit.
 * @tparam Scale The scale factor for conversion.
 * @tparam Offset The offset for conversion.
 * @tparam ErrorChecking The error checking policy.
 */
template <typename T, typename D, T Scale, T Offset = T(0), error_checking ErrorChecking = error_checking::disabled>
struct linear_scaled_unit : unit_t<T, D, ErrorChecking> {
    using unit_t<T, D, ErrorChecking>::unit_t;

    /**
     * @brief Convert from the base unit to this unit.
     *
     * @param u The base unit to convert.
     * @return The value in this unit.
     */
    static constexpr auto convert_to(const unit_t<T, D, ErrorChecking> &u, const linear_scaled_unit & /*unused*/) -> T {
        return (u.value() - Offset) / Scale;
    }

    /**
     * @brief Convert from this unit to the base unit.
     *
     * @param value The value in this unit.
     * @return The value in the base unit.
     */
    static constexpr auto convert_from(T value) -> T { return value * Scale + Offset; }
};

/**
 * @brief Helper function to create units with linear scaling.
 *
 * @tparam T The underlying numeric type.
 * @tparam D The dimension of the unit.
 * @tparam Scale The scale factor for conversion.
 * @tparam Offset The offset for conversion.
 * @tparam ErrorChecking The error checking policy.
 * @param value The value in the unit being created.
 * @return A linear_scaled_unit representing the value.
 */
template <typename T, typename D, T Scale, T Offset = T(0), error_checking ErrorChecking = error_checking::disabled>
constexpr auto make_unit(T value) {
    return linear_scaled_unit<T, D, Scale, Offset, ErrorChecking>(
        linear_scaled_unit<T, D, Scale, Offset, ErrorChecking>::convert_from(value));
}

/// @brief Length units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing meters
constexpr auto meters(T value) {
    return make_unit<T, dimensions::length, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing feet
constexpr auto feet(T value) {
    return make_unit<T, dimensions::length, T(FEET_TO_METERS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing inches
constexpr auto inches(T value) {
    return make_unit<T, dimensions::length, T(INCHES_TO_METERS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing kilometers
constexpr auto kilometers(T value) {
    return make_unit<T, dimensions::length, T(KILOMETERS_TO_METERS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing miles
constexpr auto miles(T value) {
    return make_unit<T, dimensions::length, T(MILES_TO_METERS)>(value);
}
/// @}

/// @brief Time units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing seconds
constexpr auto seconds(T value) {
    return make_unit<T, dimensions::time, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing minutes
constexpr auto minutes(T value) {
    return make_unit<T, dimensions::time, T(MINUTES_TO_SECONDS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing hours
constexpr auto hours(T value) {
    return make_unit<T, dimensions::time, T(HOURS_TO_SECONDS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing days
constexpr auto days(T value) {
    return make_unit<T, dimensions::time, T(DAYS_TO_SECONDS)>(value);
}
/// @}

/// @brief Mass units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing kilograms
constexpr auto kilograms(T value) {
    return make_unit<T, dimensions::mass, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing grams
constexpr auto grams(T value) {
    return make_unit<T, dimensions::mass, T(GRAMS_TO_KILOGRAMS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing pounds
constexpr auto pounds(T value) {
    return make_unit<T, dimensions::mass, T(POUNDS_TO_KILOGRAMS)>(value);
}
/// @}

/// @brief Temperature units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing kelvin
constexpr auto kelvin(T value) {
    return make_unit<T, dimensions::temperature, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing celsius
constexpr auto celsius(T value) {
    return make_unit<T, dimensions::temperature, T(1), T(CELSIUS_OFFSET)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing fahrenheit
constexpr auto fahrenheit(T value) {
    return make_unit<T, dimensions::temperature, T(FAHRENHEIT_SCALE), T(FAHRENHEIT_OFFSET)>(value);
}
/// @}

/// @brief Angle units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing radians
constexpr auto radians(T value) {
    return make_unit<T, dimensions::dimensionless, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing degrees
constexpr auto degrees(T value) {
    return make_unit<T, dimensions::dimensionless, T(std::numbers::pi_v<T>) / T(DEGREES_IN_HALF_CIRCLE)>(value);
}
/// @}

/// @brief Velocity units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing meters per second
constexpr auto meters_per_second(T value) {
    return make_unit<T, dimensions::velocity, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing kilometers per hour
constexpr auto kilometers_per_hour(T value) {
    return make_unit<T, dimensions::velocity, T(KMH_TO_MS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing miles per hour
constexpr auto miles_per_hour(T value) {
    return make_unit<T, dimensions::velocity, T(MPH_TO_MS)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing feet per second
constexpr auto feet_per_second(T value) {
    return make_unit<T, dimensions::velocity, T(FEET_TO_METERS)>(value);
}
/// @}

/// @brief Acceleration units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing meters per second squared
constexpr auto meters_per_second_squared(T value) {
    return make_unit<T, dimensions::acceleration, T(1)>(value);
}

template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing feet per second squared
constexpr auto feet_per_second_squared(T value) {
    return make_unit<T, dimensions::acceleration, T(FEET_TO_METERS)>(value);
}
/// @}

/// @brief Force units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing newtons
constexpr auto newtons(T value) {
    return make_unit<T, dimensions::force, T(1)>(value);
}
/// @}

/// @brief Energy units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing joules
constexpr auto joules(T value) {
    return make_unit<T, dimensions::energy, T(1)>(value);
}
/// @}

/// @brief Power units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing watts
constexpr auto watts(T value) {
    return make_unit<T, dimensions::power, T(1)>(value);
}
/// @}

/// @brief Pressure units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing pascals
constexpr auto pascals(T value) {
    return make_unit<T, dimensions::pressure, T(1)>(value);
}
/// @}

/// @brief Electric current units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing amperes
constexpr auto amperes(T value) {
    return make_unit<T, dimensions::current, T(1)>(value);
}
/// @}

/// @brief Electric potential units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing volts
constexpr auto volts(T value) {
    return make_unit<T, dimensions::voltage, T(1)>(value);
}
/// @}

/// @brief Electric resistance units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing ohms
constexpr auto ohms(T value) {
    return make_unit<T, dimensions::resistance, T(1)>(value);
}
/// @}

/// @brief Electric capacitance units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing farads
constexpr auto farads(T value) {
    return make_unit<T, dimensions::capacitance, T(1)>(value);
}
/// @}

/// @brief Magnetic flux units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing webers
constexpr auto webers(T value) {
    return make_unit<T, dimensions::magnetic_flux, T(1)>(value);
}
/// @}

/// @brief Magnetic flux density units
/// @{
template <typename T = float, error_checking ErrorChecking = error_checking::disabled>
/// @brief Create a quantity representing teslas
constexpr auto teslas(T value) {
    return make_unit<T, dimensions::magnetic_flux_density, T(1)>(value);
}
/// @}

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_HPP