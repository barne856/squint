#ifndef SQUINT_QUANTITY_UNIT_TYPES_HPP
#define SQUINT_QUANTITY_UNIT_TYPES_HPP

#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/unit.hpp"
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
inline constexpr auto POUNDSF_TO_NEWTONS = 4.4482216152605;
inline constexpr auto CELSIUS_OFFSET = 273.15;
inline constexpr auto FAHRENHEIT_SCALE = 5.0 / 9.0;
inline constexpr auto FAHRENHEIT_OFFSET = 459.67;
inline constexpr auto DEGREES_IN_HALF_CIRCLE = 180.0;
inline constexpr auto PARTICLES_PER_MOLE = 6.02214076e23;
inline constexpr auto HORSEPOWER_TO_WATTS = 745.7;
inline constexpr auto ATMOSPHERES_TO_PASCALS = 101325.0;
inline constexpr auto BAR_TO_PASCALS = 100000.0;
inline constexpr auto LITER_TO_CUBIC_METERS = 0.001;
inline constexpr auto GALLONS_TO_CUBIC_METERS = 0.00378541;
/// @}

/// @brief Length units
/// @{
template <typename T> using meters_t = unit<T, dimensions::L>;
template <typename T> using feet_t = unit<T, dimensions::L, static_cast<T>(FEET_TO_METERS)>;
template <typename T> using inches_t = unit<T, dimensions::L, static_cast<T>(INCHES_TO_METERS)>;
template <typename T> using kilometers_t = unit<T, dimensions::L, static_cast<T>(KILOMETERS_TO_METERS)>;
template <typename T> using miles_t = unit<T, dimensions::L, static_cast<T>(MILES_TO_METERS)>;
/// @}

/// @brief Time units
/// @{
template <typename T> using seconds_t = unit<T, dimensions::T>;
template <typename T> using minutes_t = unit<T, dimensions::T, static_cast<T>(MINUTES_TO_SECONDS)>;
template <typename T> using hours_t = unit<T, dimensions::T, static_cast<T>(HOURS_TO_SECONDS)>;
template <typename T> using days_t = unit<T, dimensions::T, static_cast<T>(DAYS_TO_SECONDS)>;
/// @}

/// @brief Mass units
/// @{
template <typename T> using kilograms_t = unit<T, dimensions::M>;
template <typename T> using grams_t = unit<T, dimensions::M, static_cast<T>(GRAMS_TO_KILOGRAMS)>;
template <typename T> using pounds_t = unit<T, dimensions::M, static_cast<T>(POUNDS_TO_KILOGRAMS)>;
/// @}

/// @brief Temperature units
/// @{
template <typename T> using kelvin_t = unit<T, dimensions::K>;
template <typename T> using celsius_t = unit<T, dimensions::K, T(1), static_cast<T>(CELSIUS_OFFSET)>;
template <typename T>
using fahrenheit_t = unit<T, dimensions::K, static_cast<T>(FAHRENHEIT_SCALE), static_cast<T>(FAHRENHEIT_OFFSET)>;
/// @}

/// @brief Electric current units
/// @{
template <typename T> using amperes_t = unit<T, dimensions::I>;
/// @}

/// @brief Amount of substance units
/// @{
template <typename T> using mole_t = unit<T, dimensions::N>;
template <typename T> using particles_t = unit<T, dimensions::N, static_cast<T>(PARTICLES_PER_MOLE)>;
/// @}

/// @brief Luminous intensity units
/// @{
template <typename T> using candela_t = unit<T, dimensions::J>;
/// @}

/// @brief Velocity units
/// @{
template <typename T> using meters_per_second_t = unit<T, dimensions::velocity_dim>;
template <typename T> using feet_per_second_t = unit<T, dimensions::velocity_dim, static_cast<T>(FEET_TO_METERS)>;
template <typename T>
using kilometers_per_hour_t =
    unit<T, dimensions::velocity_dim, static_cast<T>(KILOMETERS_TO_METERS / HOURS_TO_SECONDS)>;
template <typename T>
using miles_per_hour_t = unit<T, dimensions::velocity_dim, static_cast<T>(MILES_TO_METERS / HOURS_TO_SECONDS)>;
/// @}

/// @brief Acceleration units
/// @{
template <typename T> using meters_per_second_squared_t = unit<T, dimensions::acceleration_dim>;
template <typename T>
using feet_per_second_squared_t = unit<T, dimensions::acceleration_dim, static_cast<T>(FEET_TO_METERS)>;
/// @}

/// @brief Force units
/// @{
template <typename T> using newtons_t = unit<T, dimensions::force_dim>;
template <typename T> using pounds_force_t = unit<T, dimensions::force_dim, static_cast<T>(POUNDSF_TO_NEWTONS)>;
/// @}

/// @brief Energy units
/// @{
template <typename T> using joules_t = unit<T, dimensions::energy_dim>;
/// @}

/// @brief Power units
/// @{
template <typename T> using watts_t = unit<T, dimensions::power_dim>;
template <typename T> using horsepower_t = unit<T, dimensions::power_dim, static_cast<T>(HORSEPOWER_TO_WATTS)>;
/// @}

/// @brief Pressure units
/// @{
template <typename T> using pascals_t = unit<T, dimensions::pressure_dim>;
template <typename T>
using psf_t = unit<T, dimensions::pressure_dim, static_cast<T>((FEET_TO_METERS * FEET_TO_METERS) / POUNDSF_TO_NEWTONS)>;
template <typename T>
using psi_t =
    unit<T, dimensions::pressure_dim, static_cast<T>((INCHES_TO_METERS * INCHES_TO_METERS) / POUNDSF_TO_NEWTONS)>;
template <typename T> using atmospheres_t = unit<T, dimensions::pressure_dim, static_cast<T>(ATMOSPHERES_TO_PASCALS)>;
template <typename T> using bar_t = unit<T, dimensions::pressure_dim, static_cast<T>(BAR_TO_PASCALS)>;
/// @}

/// @brief Electric charge units
/// @{
template <typename T> using coulombs_t = unit<T, dimensions::charge_dim>;
/// @}

/// @brief Area units
/// @{
template <typename T> using square_meters_t = unit<T, dimensions::area_dim>;
template <typename T>
using square_feet_t = unit<T, dimensions::area_dim, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using square_inches_t = unit<T, dimensions::area_dim, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS)>;
template <typename T>
using square_miles_t = unit<T, dimensions::area_dim, static_cast<T>(MILES_TO_METERS *MILES_TO_METERS)>;
template <typename T>
using square_kilometers_t = unit<T, dimensions::area_dim, static_cast<T>(KILOMETERS_TO_METERS *KILOMETERS_TO_METERS)>;
/// @}

/// @brief Volume units
/// @{
template <typename T> using cubic_meters_t = unit<T, dimensions::volume_dim>;
template <typename T>
using cubic_feet_t = unit<T, dimensions::volume_dim, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using cubic_inches_t =
    unit<T, dimensions::volume_dim, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS *INCHES_TO_METERS)>;
template <typename T>
using cubic_miles_t =
    unit<T, dimensions::volume_dim, static_cast<T>(MILES_TO_METERS *MILES_TO_METERS *MILES_TO_METERS)>;
template <typename T>
using cubic_kilometers_t =
    unit<T, dimensions::volume_dim, static_cast<T>(KILOMETERS_TO_METERS *KILOMETERS_TO_METERS *KILOMETERS_TO_METERS)>;
template <typename T> using liters_t = unit<T, dimensions::volume_dim, static_cast<T>(LITER_TO_CUBIC_METERS)>;
template <typename T> using gallons_t = unit<T, dimensions::volume_dim, static_cast<T>(GALLONS_TO_CUBIC_METERS)>;
/// @}

/// @brief Density units
/// @{
template <typename T> using kilograms_per_cubic_meter_t = unit<T, dimensions::density_dim>;
template <typename T>
using pounds_per_cubic_foot_t =
    unit<T, dimensions::density_dim,
         static_cast<T>(POUNDS_TO_KILOGRAMS / (FEET_TO_METERS * FEET_TO_METERS * FEET_TO_METERS))>;
/// @}

/// @brief Frequency units
/// @{
template <typename T> using hertz_t = unit<T, dimensions::frequency_dim>;
/// @}

/// @brief Angle units
/// @{
template <typename T> using radians_t = unit<T, dimensions::angle_dim>;
template <typename T>
using degrees_t = unit<T, dimensions::angle_dim, static_cast<T>(std::numbers::pi_v<T> / DEGREES_IN_HALF_CIRCLE)>;
/// @}

/// @brief Angular velocity units
/// @{
template <typename T> using radians_per_second_t = unit<T, dimensions::angular_velocity_dim>;
template <typename T>
using degrees_per_second_t =
    unit<T, dimensions::angular_velocity_dim, static_cast<T>(std::numbers::pi_v<T> / DEGREES_IN_HALF_CIRCLE)>;
/// @}

/// @brief Angular acceleration units
/// @{
template <typename T> using radians_per_second_squared_t = unit<T, dimensions::angular_acceleration_dim>;
template <typename T>
using degrees_per_second_squared_t =
    unit<T, dimensions::angular_acceleration_dim, static_cast<T>(std::numbers::pi_v<T> / DEGREES_IN_HALF_CIRCLE)>;
/// @}

/// @brief Torque units
/// @{
template <typename T> using newton_meters_t = unit<T, dimensions::torque_dim>;
template <typename T>
using foot_pounds_t = unit<T, dimensions::torque_dim, static_cast<T>(FEET_TO_METERS *POUNDSF_TO_NEWTONS)>;
/// @}

/// @brief Moment of inertia units
/// @{
template <typename T> using kilogram_square_meters_t = unit<T, dimensions::moment_of_inertia_dim>;
template <typename T>
using pound_square_feet_t =
    unit<T, dimensions::moment_of_inertia_dim, static_cast<T>(POUNDS_TO_KILOGRAMS *(FEET_TO_METERS *FEET_TO_METERS))>;
/// @}

/// @brief Linear momentum units
/// @{
template <typename T> using kilogram_meters_per_second_t = unit<T, dimensions::momentum_dim>;
template <typename T>
using pound_feet_per_second_t = unit<T, dimensions::momentum_dim, static_cast<T>(POUNDS_TO_KILOGRAMS *FEET_TO_METERS)>;
/// @}

/// @brief Voltage units
/// @{
template <typename T> using volts_t = unit<T, dimensions::voltage_dim>;
/// @}

/// @brief Inductance units
/// @{
template <typename T> using henrys_t = unit<T, dimensions::inductance_dim>;
/// @}

/// @brief Capacitance units
/// @{
template <typename T> using farads_t = unit<T, dimensions::capacitance_dim>;
/// @}

/// @brief Type aliases for length quantities with float type
/// @{
using meters = meters_t<float>;
using feet = feet_t<float>;
using inches = inches_t<float>;
using kilometers = kilometers_t<float>;
using miles = miles_t<float>;
/// @}

/// @brief Type aliases for time quantities with float type
/// @{
using seconds = seconds_t<float>;
using minutes = minutes_t<float>;
using hours = hours_t<float>;
using days = days_t<float>;
/// @}

/// @brief Type aliases for mass quantities with float type
/// @{
using kilograms = kilograms_t<float>;
using grams = grams_t<float>;
using pounds = pounds_t<float>;
/// @}

/// @brief Type aliases for temperature quantities with float type
/// @{
using kelvin = kelvin_t<float>;
using celsius = celsius_t<float>;
using fahrenheit = fahrenheit_t<float>;
/// @}

/// @brief Type aliases for electric current quantities with float type
/// @{
using amperes = amperes_t<float>;
/// @}

/// @brief Type aliases for amount of substance quantities with float type
/// @{
using mole = mole_t<float>;
using particles = particles_t<float>;
/// @}

/// @brief Type aliases for luminous intensity quantities with float type
/// @{
using candela = candela_t<float>;
/// @}

/// @brief Type aliases for velocity quantities with float type
/// @{
using meters_per_second = meters_per_second_t<float>;
using feet_per_second = feet_per_second_t<float>;
using kilometers_per_hour = kilometers_per_hour_t<float>;
using miles_per_hour = miles_per_hour_t<float>;
/// @}

/// @brief Type aliases for acceleration quantities with float type
/// @{
using meters_per_second_squared = meters_per_second_squared_t<float>;
using feet_per_second_squared = feet_per_second_squared_t<float>;
/// @}

/// @brief Type aliases for force quantities with float type
/// @{
using newtons = newtons_t<float>;
using pounds_force = pounds_force_t<float>;
/// @}

/// @brief Type aliases for energy quantities with float type
/// @{
using joules = joules_t<float>;
/// @}

/// @brief Type aliases for power quantities with float type
/// @{
using watts = watts_t<float>;
using horsepower = horsepower_t<float>;
/// @}

/// @brief Type aliases for pressure quantities with float type
/// @{
using pascals = pascals_t<float>;
using psf = psf_t<float>;
using psi = psi_t<float>;
using atmospheres = atmospheres_t<float>;
using bar = bar_t<float>;
/// @}

/// @brief Type aliases for electric charge quantities with float type
/// @{
using coulombs = coulombs_t<float>;
/// @}

/// @brief Type aliases for area quantities with float type
/// @{
using square_meters = square_meters_t<float>;
using square_feet = square_feet_t<float>;
using square_inches = square_inches_t<float>;
using square_miles = square_miles_t<float>;
using square_kilometers = square_kilometers_t<float>;
/// @}

/// @brief Type aliases for volume quantities with float type
/// @{
using cubic_meters = cubic_meters_t<float>;
using cubic_feet = cubic_feet_t<float>;
using cubic_inches = cubic_inches_t<float>;
using cubic_miles = cubic_miles_t<float>;
using cubic_kilometers = cubic_kilometers_t<float>;
using liters = liters_t<float>;
using gallons = gallons_t<float>;
/// @}

/// @brief Type aliases for density quantities with float type
/// @{
using kilograms_per_cubic_meter = kilograms_per_cubic_meter_t<float>;
using pounds_per_cubic_foot = pounds_per_cubic_foot_t<float>;
/// @}

/// @brief Type aliases for frequency quantities with float type
/// @{
using hertz = hertz_t<float>;
/// @}

/// @brief Type aliases for angle quantities with float type
/// @{
using radians = radians_t<float>;
using degrees = degrees_t<float>;
/// @}

/// @brief Type aliases for angular velocity quantities with float type
/// @{
using radians_per_second = radians_per_second_t<float>;
using degrees_per_second = degrees_per_second_t<float>;
/// @}

/// @brief Type aliases for angular acceleration quantities with float type
/// @{
using radians_per_second_squared = radians_per_second_squared_t<float>;
using degrees_per_second_squared = degrees_per_second_squared_t<float>;
/// @}

/// @brief Type aliases for torque quantities with float type
/// @{
using newton_meters = newton_meters_t<float>;
using foot_pounds = foot_pounds_t<float>;
/// @}

/// @brief Type aliases for moment of inertia quantities with float type
/// @{
using kilogram_square_meters = kilogram_square_meters_t<float>;
using pound_square_feet = pound_square_feet_t<float>;
/// @}

/// @brief Type aliases for linear momentum quantities with float type
/// @{
using kilogram_meters_per_second = kilogram_meters_per_second_t<float>;
using pound_feet_per_second = pound_feet_per_second_t<float>;
/// @}

/// @brief Type aliases for voltage quantities with float type
/// @{
using volts = volts_t<float>;
/// @}

/// @brief Type aliases for inductance quantities with float type
/// @{
using henrys = henrys_t<float>;
/// @}

/// @brief Type aliases for capacitance quantities with float type
/// @{
using farads = farads_t<float>;
/// @}

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_TYPES_HPP