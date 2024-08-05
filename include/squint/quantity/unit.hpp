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
 * @brief Base unit struct for all physical quantities.
 *
 * @tparam T The underlying numeric type (e.g., float, double).
 * @tparam D The dimension of the unit.
 * @tparam Scale The scale factor for conversion.
 * @tparam Offset The offset for conversion.
 * @tparam ErrorChecking The error checking policy.
 */
template <typename T, typename D, T Scale = T(1), T Offset = T(0),
          error_checking ErrorChecking = error_checking::disabled>
struct unit : quantity<T, D, ErrorChecking> {
    using base_type = quantity<T, D, ErrorChecking>;
    using base_type::base_type; // Inherit constructors

    static constexpr T scale = Scale;
    static constexpr T offset = Offset;

    // Constructor from value in this unit
    constexpr explicit unit(T value) : base_type((value + offset) * scale) {}

    // Convert to value in this unit
    [[nodiscard]] constexpr auto unit_value() const -> T { return this->base_type::value() / scale - offset; }

    // Convert from base unit to this unit
    static constexpr auto convert_to(const base_type &q) -> T { return q.value() / scale - offset; }

    // Convert from this unit to base unit
    static constexpr auto convert_from(T value) -> base_type { return base_type((value + offset) * scale); }
};

/// @brief Dimensionless units
/// @{
template <typename T> using dimensionless_t = unit<T, dimensions::dimensionless>;
/// @}

/// @brief Length units
/// @{
template <typename T> using meters_t = unit<T, dimensions::length>;
template <typename T> using feet_t = unit<T, dimensions::length, static_cast<T>(FEET_TO_METERS)>;
template <typename T> using inches_t = unit<T, dimensions::length, static_cast<T>(INCHES_TO_METERS)>;
template <typename T> using kilometers_t = unit<T, dimensions::length, static_cast<T>(KILOMETERS_TO_METERS)>;
template <typename T> using miles_t = unit<T, dimensions::length, static_cast<T>(MILES_TO_METERS)>;
/// @}

/// @brief Area units
/// @{
template <typename T> using square_meters_t = unit<T, dimensions::area>;
template <typename T> using square_feet_t = unit<T, dimensions::area, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using square_inches_t = unit<T, dimensions::area, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS)>;
template <typename T>
using square_kilometers_t = unit<T, dimensions::area, static_cast<T>(KILOMETERS_TO_METERS *KILOMETERS_TO_METERS)>;
template <typename T>
using square_miles_t = unit<T, dimensions::area, static_cast<T>(MILES_TO_METERS *MILES_TO_METERS)>;
/// @}

/// @brief Volume units
/// @{
template <typename T> using cubic_meters_t = unit<T, dimensions::volume>;
template <typename T>
using cubic_feet_t = unit<T, dimensions::volume, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using cubic_inches_t =
    unit<T, dimensions::volume, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS *INCHES_TO_METERS)>;
template <typename T>
using cubic_kilometers_t =
    unit<T, dimensions::volume, static_cast<T>(KILOMETERS_TO_METERS *KILOMETERS_TO_METERS *KILOMETERS_TO_METERS)>;
template <typename T>
using cubic_miles_t = unit<T, dimensions::volume, static_cast<T>(MILES_TO_METERS *MILES_TO_METERS *MILES_TO_METERS)>;
/// @}

/// @brief Time units
/// @{
template <typename T> using seconds_t = unit<T, dimensions::time>;
template <typename T> using minutes_t = unit<T, dimensions::time, static_cast<T>(MINUTES_TO_SECONDS)>;
template <typename T> using hours_t = unit<T, dimensions::time, static_cast<T>(HOURS_TO_SECONDS)>;
template <typename T> using days_t = unit<T, dimensions::time, static_cast<T>(DAYS_TO_SECONDS)>;
/// @}

/// @brief Mass units
/// @{
template <typename T> using kilograms_t = unit<T, dimensions::mass>;
template <typename T> using grams_t = unit<T, dimensions::mass, static_cast<T>(GRAMS_TO_KILOGRAMS)>;
template <typename T> using pounds_t = unit<T, dimensions::mass, static_cast<T>(POUNDS_TO_KILOGRAMS)>;
/// @}

/// @brief Temperature units
/// @{
template <typename T> using kelvin_t = unit<T, dimensions::temperature>;
template <typename T> using celsius_t = unit<T, dimensions::temperature, T(1), static_cast<T>(CELSIUS_OFFSET)>;
template <typename T>
using fahrenheit_t =
    unit<T, dimensions::temperature, static_cast<T>(FAHRENHEIT_SCALE), static_cast<T>(FAHRENHEIT_OFFSET)>;
/// @}

/// @brief Angle units
/// @{
template <typename T> using radians_t = unit<T, dimensions::dimensionless>;
template <typename T>
using degrees_t = unit<T, dimensions::dimensionless, static_cast<T>(std::numbers::pi / DEGREES_IN_HALF_CIRCLE)>;
/// @}

/// @brief Velocity units
/// @{
template <typename T> using meters_per_second_t = unit<T, dimensions::velocity>;
template <typename T> using kilometers_per_hour_t = unit<T, dimensions::velocity, static_cast<T>(KMH_TO_MS)>;
template <typename T> using miles_per_hour_t = unit<T, dimensions::velocity, static_cast<T>(MPH_TO_MS)>;
template <typename T> using feet_per_second_t = unit<T, dimensions::velocity, static_cast<T>(FEET_TO_METERS)>;
/// @}

/// @brief Acceleration units
/// @{
template <typename T> using meters_per_second_squared_t = unit<T, dimensions::acceleration>;
template <typename T>
using feet_per_second_squared_t = unit<T, dimensions::acceleration, static_cast<T>(FEET_TO_METERS)>;
/// @}

/// @brief Force units
/// @{
template <typename T> using newtons_t = unit<T, dimensions::force>;
/// @}

/// @brief Energy units
/// @{
template <typename T> using joules_t = unit<T, dimensions::energy>;
/// @}

/// @brief Power units
/// @{
template <typename T> using watts_t = unit<T, dimensions::power>;
/// @}

/// @brief Pressure units
/// @{
template <typename T> using pascals_t = unit<T, dimensions::pressure>;
/// @}

/// @brief Electric current units
/// @{
template <typename T> using amperes_t = unit<T, dimensions::current>;
/// @}

/// @brief Electric potential units
/// @{
template <typename T> using volts_t = unit<T, dimensions::voltage>;
/// @}

/// @brief Electric resistance units
/// @{
template <typename T> using ohms_t = unit<T, dimensions::resistance>;
/// @}

/// @brief Electric capacitance units
/// @{
template <typename T> using farads_t = unit<T, dimensions::capacitance>;
/// @}

/// @brief Magnetic flux units
/// @{
template <typename T> using webers_t = unit<T, dimensions::magnetic_flux>;
/// @}

/// @brief Magnetic flux density units
/// @{
template <typename T> using teslas_t = unit<T, dimensions::magnetic_flux_density>;
/// @}

/**
 * @brief Generic conversion function between units.
 *
 * @tparam ToUnit The unit type to convert to (template template parameter).
 * @tparam FromUnit The concrete unit type to convert from.
 * @param q The quantity to convert.
 * @return A new unit object of the target unit type.
 */
template <template <typename> class ToUnit, typename FromUnit> constexpr auto convert_to(const FromUnit &q) {
    using T = typename FromUnit::value_type;
    using ToUnitType = ToUnit<T>;
    return ToUnitType(ToUnitType::convert_to(q));
}

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_HPP