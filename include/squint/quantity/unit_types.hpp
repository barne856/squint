/**
 * @file unit_types.hpp
 * @brief Defines unit types for various physical quantities.
 *
 * This file provides a comprehensive set of unit types for physical quantities,
 * including conversion factors between different units of the same quantity.
 */

#ifndef SQUINT_QUANTITY_UNIT_TYPES_HPP
#define SQUINT_QUANTITY_UNIT_TYPES_HPP

#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/unit.hpp"
#include <numbers>

namespace squint::units {
// NOLINTBEGIN
/// @brief Conversion constants for unit conversions
/// @{
inline constexpr auto FEET_TO_METERS = 0.3048;
inline constexpr auto INCHES_TO_METERS = 0.0254;
inline constexpr auto KILOMETERS_TO_METERS = 1000.0;
inline constexpr auto MILES_TO_METERS = 1609.344;
inline constexpr auto NAUTICAL_MILES_TO_METERS = 1852.0;
inline constexpr auto LIGHT_YEARS_TO_METERS = 9.4607304725808e15;

inline constexpr auto MINUTES_TO_SECONDS = 60.0;
inline constexpr auto HOURS_TO_SECONDS = 3600.0;
inline constexpr auto DAYS_TO_SECONDS = 86400.0;
inline constexpr auto YEARS_TO_SECONDS = 31557600.0;

inline constexpr auto GRAMS_TO_KILOGRAMS = 0.001;
inline constexpr auto POUNDS_TO_KILOGRAMS = 0.45359237;
inline constexpr auto OUNCES_TO_KILOGRAMS = 0.028349523125;
inline constexpr auto TONNES_TO_KILOGRAMS = 1000.0;

inline constexpr auto CELSIUS_OFFSET = 273.15;
inline constexpr auto FAHRENHEIT_SCALE = 5.0 / 9.0;
inline constexpr auto FAHRENHEIT_OFFSET = 459.67;

inline constexpr auto DEGREES_TO_RADIANS = std::numbers::pi / 180.0;
inline constexpr auto ARCMINUTES_TO_RADIANS = DEGREES_TO_RADIANS / 60.0;
inline constexpr auto ARCSECONDS_TO_RADIANS = DEGREES_TO_RADIANS / 3600.0;

inline constexpr auto CALORIES_TO_JOULES = 4.184;
inline constexpr auto BTU_TO_JOULES = 1055.06;
inline constexpr auto KILOWATT_HOURS_TO_JOULES = 3.6e6;
inline constexpr auto ELECTRON_VOLTS_TO_JOULES = 1.602176634e-19;

inline constexpr auto HORSEPOWER_TO_WATTS = 745.7;

inline constexpr auto ATMOSPHERES_TO_PASCALS = 101325.0;
inline constexpr auto BAR_TO_PASCALS = 100000.0;
inline constexpr auto MMHG_TO_PASCALS = 133.322;
inline constexpr auto PSI_TO_PASCALS = 6894.75729;

inline constexpr auto LITERS_TO_CUBIC_METERS = 0.001;
inline constexpr auto GALLONS_TO_CUBIC_METERS = 0.00378541;

inline constexpr auto POISE_TO_PASCAL_SECONDS = 0.1;

inline constexpr auto GAUSS_TO_TESLA = 1e-4;

inline constexpr auto STATCOULOMB_TO_COULOMB = 3.335641e-10;

inline constexpr auto STATFARAD_TO_FARAD = 1.112650e-12;

inline constexpr auto STATHENRY_TO_HENRY = 8.987552e11;

inline constexpr auto STATOHM_TO_OHM = 8.987552e11;

inline constexpr auto STATVOLT_TO_VOLT = 299.792458;

inline constexpr auto DEBYE_TO_COULOMB_METER = 3.33564e-30;

inline constexpr auto DYNE_TO_NEWTON = 1e-5;

inline constexpr auto ERG_TO_JOULE = 1e-7;

inline constexpr auto GILBERT_TO_AMPERE_TURN = 0.795774715459;

inline constexpr auto STILB_TO_CANDELA_PER_SQUARE_METER = 1e4;

inline constexpr auto PHOT_TO_LUX = 1e4;

inline constexpr auto CURIE_TO_BECQUEREL = 3.7e10;

inline constexpr auto ROENTGEN_TO_COULOMB_PER_KILOGRAM = 2.58e-4;

inline constexpr auto RAD_TO_GRAY = 0.01;

inline constexpr auto REM_TO_SIEVERT = 0.01;
/// @}

/// @brief Length units
/// @{
template <typename T> using meters_t = unit<T, dimensions::L>;
template <typename T> using feet_t = unit<T, dimensions::L, static_cast<T>(FEET_TO_METERS)>;
template <typename T> using inches_t = unit<T, dimensions::L, static_cast<T>(INCHES_TO_METERS)>;
template <typename T> using kilometers_t = unit<T, dimensions::L, static_cast<T>(KILOMETERS_TO_METERS)>;
template <typename T> using miles_t = unit<T, dimensions::L, static_cast<T>(MILES_TO_METERS)>;
template <typename T> using nautical_miles_t = unit<T, dimensions::L, static_cast<T>(NAUTICAL_MILES_TO_METERS)>;
template <typename T> using light_years_t = unit<T, dimensions::L, static_cast<T>(LIGHT_YEARS_TO_METERS)>;
/// @}

/// @brief Time units
/// @{
template <typename T> using seconds_t = unit<T, dimensions::T>;
template <typename T> using minutes_t = unit<T, dimensions::T, static_cast<T>(MINUTES_TO_SECONDS)>;
template <typename T> using hours_t = unit<T, dimensions::T, static_cast<T>(HOURS_TO_SECONDS)>;
template <typename T> using days_t = unit<T, dimensions::T, static_cast<T>(DAYS_TO_SECONDS)>;
template <typename T> using years_t = unit<T, dimensions::T, static_cast<T>(YEARS_TO_SECONDS)>;
/// @}

/// @brief Mass units
/// @{
template <typename T> using kilograms_t = unit<T, dimensions::M>;
template <typename T> using grams_t = unit<T, dimensions::M, static_cast<T>(GRAMS_TO_KILOGRAMS)>;
template <typename T> using pounds_t = unit<T, dimensions::M, static_cast<T>(POUNDS_TO_KILOGRAMS)>;
template <typename T> using ounces_t = unit<T, dimensions::M, static_cast<T>(OUNCES_TO_KILOGRAMS)>;
template <typename T> using tonnes_t = unit<T, dimensions::M, static_cast<T>(TONNES_TO_KILOGRAMS)>;
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
/// @}

/// @brief Luminous intensity units
/// @{
template <typename T> using candela_t = unit<T, dimensions::J>;
/// @}

/// @brief Angle units
/// @{
template <typename T> using radians_t = unit<T, dimensions::angle_dim>;
template <typename T> using degrees_t = unit<T, dimensions::angle_dim, static_cast<T>(DEGREES_TO_RADIANS)>;
template <typename T> using arcminutes_t = unit<T, dimensions::angle_dim, static_cast<T>(ARCMINUTES_TO_RADIANS)>;
template <typename T> using arcseconds_t = unit<T, dimensions::angle_dim, static_cast<T>(ARCSECONDS_TO_RADIANS)>;
/// @}

/// @brief Velocity units
/// @{
template <typename T> using meters_per_second_t = unit<T, dimensions::velocity_dim>;
template <typename T> using kilometers_per_hour_t = unit<T, dimensions::velocity_dim, static_cast<T>(1.0 / 3.6)>;
template <typename T>
using miles_per_hour_t = unit<T, dimensions::velocity_dim, static_cast<T>(MILES_TO_METERS / 3600.0)>;
template <typename T> using feet_per_second_t = unit<T, dimensions::velocity_dim, static_cast<T>(FEET_TO_METERS)>;
template <typename T>
using knots_t = unit<T, dimensions::velocity_dim, static_cast<T>(NAUTICAL_MILES_TO_METERS / 3600.0)>;
/// @}

/// @brief Acceleration units
/// @{
template <typename T> using meters_per_second_squared_t = unit<T, dimensions::acceleration_dim>;
template <typename T>
using feet_per_second_squared_t = unit<T, dimensions::acceleration_dim, static_cast<T>(FEET_TO_METERS)>;
template <typename T> using standard_gravity_t = unit<T, dimensions::acceleration_dim, static_cast<T>(9.80665)>;
/// @}

/// @brief Force units
/// @{
template <typename T> using newtons_t = unit<T, dimensions::force_dim>;
template <typename T>
using pounds_force_t = unit<T, dimensions::force_dim, static_cast<T>(POUNDS_TO_KILOGRAMS * 9.80665)>;
template <typename T> using dynes_t = unit<T, dimensions::force_dim, static_cast<T>(DYNE_TO_NEWTON)>;
/// @}

/// @brief Energy units
/// @{
template <typename T> using joules_t = unit<T, dimensions::energy_dim>;
template <typename T> using calories_t = unit<T, dimensions::energy_dim, static_cast<T>(CALORIES_TO_JOULES)>;
template <typename T>
using kilocalories_t = unit<T, dimensions::energy_dim, static_cast<T>(1000.0 * CALORIES_TO_JOULES)>;
template <typename T> using btu_t = unit<T, dimensions::energy_dim, static_cast<T>(BTU_TO_JOULES)>;
template <typename T>
using kilowatt_hours_t = unit<T, dimensions::energy_dim, static_cast<T>(KILOWATT_HOURS_TO_JOULES)>;
template <typename T>
using electron_volts_t = unit<T, dimensions::energy_dim, static_cast<T>(ELECTRON_VOLTS_TO_JOULES)>;
template <typename T> using ergs_t = unit<T, dimensions::energy_dim, static_cast<T>(ERG_TO_JOULE)>;
/// @}

/// @brief Power units
/// @{
template <typename T> using watts_t = unit<T, dimensions::power_dim>;
template <typename T> using horsepower_t = unit<T, dimensions::power_dim, static_cast<T>(HORSEPOWER_TO_WATTS)>;
/// @}

/// @brief Pressure units
/// @{
template <typename T> using pascals_t = unit<T, dimensions::pressure_dim>;
template <typename T> using atmospheres_t = unit<T, dimensions::pressure_dim, static_cast<T>(ATMOSPHERES_TO_PASCALS)>;
template <typename T> using bars_t = unit<T, dimensions::pressure_dim, static_cast<T>(BAR_TO_PASCALS)>;
template <typename T>
using millimeters_of_mercury_t = unit<T, dimensions::pressure_dim, static_cast<T>(MMHG_TO_PASCALS)>;
template <typename T>
using pounds_per_square_inch_t = unit<T, dimensions::pressure_dim, static_cast<T>(PSI_TO_PASCALS)>;
/// @}

/// @brief Electric charge units
/// @{
template <typename T> using coulombs_t = unit<T, dimensions::charge_dim>;
template <typename T> using ampere_hours_t = unit<T, dimensions::charge_dim, static_cast<T>(3600.0)>;
template <typename T> using statcoulombs_t = unit<T, dimensions::charge_dim, static_cast<T>(STATCOULOMB_TO_COULOMB)>;
/// @}

/// @brief Electric potential units
/// @{
template <typename T> using volts_t = unit<T, dimensions::voltage_dim>;
template <typename T> using statvolts_t = unit<T, dimensions::voltage_dim, static_cast<T>(STATVOLT_TO_VOLT)>;
/// @}

/// @brief Capacitance units
/// @{
template <typename T> using farads_t = unit<T, dimensions::capacitance_dim>;
template <typename T> using statfarads_t = unit<T, dimensions::capacitance_dim, static_cast<T>(STATFARAD_TO_FARAD)>;
/// @}

/// @brief Resistance units
/// @{
template <typename T> using ohms_t = unit<T, dimensions::resistance_dim>;
template <typename T> using statohms_t = unit<T, dimensions::resistance_dim, static_cast<T>(STATOHM_TO_OHM)>;
/// @}

/// @brief Conductance units
/// @{
template <typename T> using siemens_t = unit<T, dimensions::conductance_dim>;
/// @}

/// @brief Magnetic flux units
/// @{
template <typename T> using webers_t = unit<T, dimensions::magnetic_flux_dim>;
template <typename T> using maxwells_t = unit<T, dimensions::magnetic_flux_dim, static_cast<T>(1e-8)>;
/// @}

/// @brief Magnetic flux density units
/// @{
template <typename T> using teslas_t = unit<T, dimensions::magnetic_flux_density_dim>;
template <typename T> using gauss_t = unit<T, dimensions::magnetic_flux_density_dim, static_cast<T>(GAUSS_TO_TESLA)>;
/// @}

/// @brief Inductance units
/// @{
template <typename T> using henries_t = unit<T, dimensions::inductance_dim>;
template <typename T> using stathenries_t = unit<T, dimensions::inductance_dim, static_cast<T>(STATHENRY_TO_HENRY)>;
/// @}

/// @brief Area units
/// @{
template <typename T> using square_meters_t = unit<T, dimensions::area_dim>;
template <typename T>
using square_feet_t = unit<T, dimensions::area_dim, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using square_inches_t = unit<T, dimensions::area_dim, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS)>;
template <typename T>
using square_kilometers_t = unit<T, dimensions::area_dim, static_cast<T>(KILOMETERS_TO_METERS *KILOMETERS_TO_METERS)>;
template <typename T>
using square_miles_t = unit<T, dimensions::area_dim, static_cast<T>(MILES_TO_METERS *MILES_TO_METERS)>;
template <typename T> using hectares_t = unit<T, dimensions::area_dim, static_cast<T>(10000.0)>;
template <typename T> using acres_t = unit<T, dimensions::area_dim, static_cast<T>(4046.8564224)>;
/// @}

/// @brief Volume units
/// @{
template <typename T> using cubic_meters_t = unit<T, dimensions::volume_dim>;
template <typename T> using liters_t = unit<T, dimensions::volume_dim, static_cast<T>(LITERS_TO_CUBIC_METERS)>;
template <typename T> using gallons_t = unit<T, dimensions::volume_dim, static_cast<T>(GALLONS_TO_CUBIC_METERS)>;
template <typename T>
using cubic_feet_t = unit<T, dimensions::volume_dim, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS *FEET_TO_METERS)>;
template <typename T>
using cubic_inches_t =
    unit<T, dimensions::volume_dim, static_cast<T>(INCHES_TO_METERS *INCHES_TO_METERS *INCHES_TO_METERS)>;
/// @}

/// @brief Density units
/// @{
template <typename T> using kilograms_per_cubic_meter_t = unit<T, dimensions::density_dim>;
template <typename T> using grams_per_cubic_centimeter_t = unit<T, dimensions::density_dim, static_cast<T>(1000.0)>;
template <typename T>
using pounds_per_cubic_foot_t =
    unit<T, dimensions::density_dim,
         static_cast<T>(POUNDS_TO_KILOGRAMS / (FEET_TO_METERS * FEET_TO_METERS * FEET_TO_METERS))>;
/// @}

/// @brief Frequency units
/// @{
template <typename T> using hertz_t = unit<T, dimensions::frequency_dim>;
template <typename T> using kilohertz_t = unit<T, dimensions::frequency_dim, static_cast<T>(1000.0)>;
template <typename T> using megahertz_t = unit<T, dimensions::frequency_dim, static_cast<T>(1e6)>;
template <typename T> using gigahertz_t = unit<T, dimensions::frequency_dim, static_cast<T>(1e9)>;
/// @}

/// @brief Angular velocity units
/// @{
template <typename T> using radians_per_second_t = unit<T, dimensions::angular_velocity_dim>;
template <typename T>
using degrees_per_second_t = unit<T, dimensions::angular_velocity_dim, static_cast<T>(DEGREES_TO_RADIANS)>;
template <typename T>
using revolutions_per_minute_t =
    unit<T, dimensions::angular_velocity_dim, static_cast<T>(2.0 * std::numbers::pi / 60.0)>;
/// @}

/// @brief Angular acceleration units
/// @{
template <typename T> using radians_per_second_squared_t = unit<T, dimensions::angular_acceleration_dim>;
template <typename T>
using degrees_per_second_squared_t = unit<T, dimensions::angular_acceleration_dim, static_cast<T>(DEGREES_TO_RADIANS)>;
/// @}

/// @brief Torque units
/// @{
template <typename T> using newton_meters_t = unit<T, dimensions::torque_dim>;
template <typename T>
using foot_pounds_t = unit<T, dimensions::torque_dim, static_cast<T>(FEET_TO_METERS *POUNDS_TO_KILOGRAMS * 9.80665)>;
/// @}

/// @brief Moment of inertia units
/// @{
template <typename T> using kilogram_square_meters_t = unit<T, dimensions::moment_of_inertia_dim>;
/// @}

/// @brief Dynamic viscosity units
/// @{
template <typename T> using pascal_seconds_t = unit<T, dimensions::dynamic_viscosity_dim>;
template <typename T>
using poise_t = unit<T, dimensions::dynamic_viscosity_dim, static_cast<T>(POISE_TO_PASCAL_SECONDS)>;
/// @}

/// @brief Kinematic viscosity units
/// @{
template <typename T> using square_meters_per_second_t = unit<T, dimensions::kinematic_viscosity_dim>;
template <typename T> using stokes_t = unit<T, dimensions::kinematic_viscosity_dim, static_cast<T>(1e-4)>;
/// @}

/// @brief Surface tension units
/// @{
template <typename T> using newtons_per_meter_t = unit<T, dimensions::surface_tension_dim>;
template <typename T> using dynes_per_centimeter_t = unit<T, dimensions::surface_tension_dim, static_cast<T>(1e-3)>;
/// @}

/// @brief Specific heat capacity units
/// @{
template <typename T> using joules_per_kilogram_kelvin_t = unit<T, dimensions::specific_heat_capacity_dim>;
template <typename T>
using calories_per_gram_celsius_t = unit<T, dimensions::specific_heat_capacity_dim, static_cast<T>(4184.0)>;
/// @}

/// @brief Thermal conductivity units
/// @{
template <typename T> using watts_per_meter_kelvin_t = unit<T, dimensions::thermal_conductivity_dim>;
/// @}

/// @brief Electric field strength units
/// @{
template <typename T> using volts_per_meter_t = unit<T, dimensions::electric_field_strength_dim>;
/// @}

/// @brief Permittivity units
/// @{
template <typename T> using farads_per_meter_t = unit<T, dimensions::permittivity_dim>;
/// @}

/// @brief Permeability units
/// @{
template <typename T> using henries_per_meter_t = unit<T, dimensions::permeability_dim>;
/// @}

/// @brief Absorbed dose units
/// @{
template <typename T> using grays_t = unit<T, dimensions::absorbed_dose_dim>;
template <typename T> using rads_t = unit<T, dimensions::absorbed_dose_dim, static_cast<T>(RAD_TO_GRAY)>;
/// @}

/// @brief Equivalent dose units
/// @{
template <typename T> using sieverts_t = unit<T, dimensions::equivalent_dose_dim>;
template <typename T> using rems_t = unit<T, dimensions::equivalent_dose_dim, static_cast<T>(REM_TO_SIEVERT)>;
/// @}

/// @brief Catalytic activity units
/// @{
template <typename T> using katal_t = unit<T, dimensions::catalytic_activity_dim>;
template <typename T> using enzyme_unit_t = unit<T, dimensions::catalytic_activity_dim, static_cast<T>(1.0 / 60.0)>;
/// @}

/// @brief Concentration units
/// @{
template <typename T> using moles_per_cubic_meter_t = unit<T, dimensions::concentration_dim>;
template <typename T> using moles_per_liter_t = unit<T, dimensions::concentration_dim, static_cast<T>(1000.0)>;
/// @}

/// @brief Molality units
/// @{
template <typename T> using moles_per_kilogram_t = unit<T, dimensions::molality_dim>;
/// @}

/// @brief Molar mass units
/// @{
template <typename T> using kilograms_per_mole_t = unit<T, dimensions::molar_mass_dim>;
template <typename T> using grams_per_mole_t = unit<T, dimensions::molar_mass_dim, static_cast<T>(0.001)>;
/// @}

/// @brief Luminous flux units
/// @{
template <typename T> using lumens_t = unit<T, dimensions::luminous_flux_dim>;
/// @}

/// @brief Illuminance units
/// @{
template <typename T> using lux_t = unit<T, dimensions::illuminance_dim>;
template <typename T> using foot_candles_t = unit<T, dimensions::illuminance_dim, static_cast<T>(10.7639)>;
/// @}

/// @brief Luminous energy units
/// @{
template <typename T> using lumen_seconds_t = unit<T, dimensions::luminous_energy_dim>;
/// @}

/// @brief Luminous exposure units
/// @{
template <typename T> using lux_seconds_t = unit<T, dimensions::luminous_exposure_dim>;
/// @}

/// @brief Radioactivity units
/// @{
template <typename T> using becquerels_t = unit<T, dimensions::radioactivity_dim>;
template <typename T> using curies_t = unit<T, dimensions::radioactivity_dim, static_cast<T>(CURIE_TO_BECQUEREL)>;
/// @}

/// @brief Flow rate units
/// @{
template <typename T> using cubic_meters_per_second_t = unit<T, dimensions::flow_dim>;
template <typename T> using liters_per_second_t = unit<T, dimensions::flow_dim, static_cast<T>(0.001)>;
template <typename T>
using gallons_per_minute_t = unit<T, dimensions::flow_dim, static_cast<T>(GALLONS_TO_CUBIC_METERS / 60.0)>;
/// @}

/// @brief Type aliases for length quantities with float type
/// @{
using meters = meters_t<float>;
using feet = feet_t<float>;
using inches = inches_t<float>;
using kilometers = kilometers_t<float>;
using miles = miles_t<float>;
using nautical_miles = nautical_miles_t<float>;
using light_years = light_years_t<float>;
/// @}

/// @brief Type aliases for time quantities with float type
/// @{
using seconds = seconds_t<float>;
using minutes = minutes_t<float>;
using hours = hours_t<float>;
using days = days_t<float>;
using years = years_t<float>;
/// @}

/// @brief Type aliases for mass quantities with float type
/// @{
using kilograms = kilograms_t<float>;
using grams = grams_t<float>;
using pounds = pounds_t<float>;
using ounces = ounces_t<float>;
using tonnes = tonnes_t<float>;
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
/// @}

/// @brief Type aliases for luminous intensity quantities with float type
/// @{
using candela = candela_t<float>;
/// @}

/// @brief Type aliases for angle quantities with float type
/// @{
using radians = radians_t<float>;
using degrees = degrees_t<float>;
using arcminutes = arcminutes_t<float>;
using arcseconds = arcseconds_t<float>;
/// @}

/// @brief Type aliases for velocity quantities with float type
/// @{
using meters_per_second = meters_per_second_t<float>;
using kilometers_per_hour = kilometers_per_hour_t<float>;
using miles_per_hour = miles_per_hour_t<float>;
using feet_per_second = feet_per_second_t<float>;
using knots = knots_t<float>;
/// @}

/// @brief Type aliases for acceleration quantities with float type
/// @{
using meters_per_second_squared = meters_per_second_squared_t<float>;
using feet_per_second_squared = feet_per_second_squared_t<float>;
using standard_gravity = standard_gravity_t<float>;
/// @}

/// @brief Type aliases for force quantities with float type
/// @{
using newtons = newtons_t<float>;
using pounds_force = pounds_force_t<float>;
using dynes = dynes_t<float>;
/// @}

/// @brief Type aliases for energy quantities with float type
/// @{
using joules = joules_t<float>;
using calories = calories_t<float>;
using kilocalories = kilocalories_t<float>;
using btu = btu_t<float>;
using kilowatt_hours = kilowatt_hours_t<float>;
using electron_volts = electron_volts_t<float>;
using ergs = ergs_t<float>;
/// @}

/// @brief Type aliases for power quantities with float type
/// @{
using watts = watts_t<float>;
using horsepower = horsepower_t<float>;
/// @}

/// @brief Type aliases for pressure quantities with float type
/// @{
using pascals = pascals_t<float>;
using atmospheres = atmospheres_t<float>;
using bars = bars_t<float>;
using millimeters_of_mercury = millimeters_of_mercury_t<float>;
using pounds_per_square_inch = pounds_per_square_inch_t<float>;
/// @}

/// @brief Type aliases for electric charge quantities with float type
/// @{
using coulombs = coulombs_t<float>;
using ampere_hours = ampere_hours_t<float>;
using statcoulombs = statcoulombs_t<float>;
/// @}

/// @brief Type aliases for electric potential quantities with float type
/// @{
using volts = volts_t<float>;
using statvolts = statvolts_t<float>;
/// @}

/// @brief Type aliases for capacitance quantities with float type
/// @{
using farads = farads_t<float>;
using statfarads = statfarads_t<float>;
/// @}

/// @brief Type aliases for resistance quantities with float type
/// @{
using ohms = ohms_t<float>;
using statohms = statohms_t<float>;
/// @}

/// @brief Type aliases for conductance quantities with float type
/// @{
using siemens = siemens_t<float>;
/// @}

/// @brief Type aliases for magnetic flux quantities with float type
/// @{
using webers = webers_t<float>;
using maxwells = maxwells_t<float>;
/// @}

/// @brief Type aliases for magnetic flux density quantities with float type
/// @{
using teslas = teslas_t<float>;
using gauss = gauss_t<float>;
/// @}

/// @brief Type aliases for inductance quantities with float type
/// @{
using henries = henries_t<float>;
using stathenries = stathenries_t<float>;
/// @}

/// @brief Type aliases for area quantities with float type
/// @{
using square_meters = square_meters_t<float>;
using square_feet = square_feet_t<float>;
using square_inches = square_inches_t<float>;
using square_kilometers = square_kilometers_t<float>;
using square_miles = square_miles_t<float>;
using hectares = hectares_t<float>;
using acres = acres_t<float>;
/// @}

/// @brief Type aliases for volume quantities with float type
/// @{
using cubic_meters = cubic_meters_t<float>;
using liters = liters_t<float>;
using gallons = gallons_t<float>;
using cubic_feet = cubic_feet_t<float>;
using cubic_inches = cubic_inches_t<float>;
/// @}

/// @brief Type aliases for density quantities with float type
/// @{
using kilograms_per_cubic_meter = kilograms_per_cubic_meter_t<float>;
using grams_per_cubic_centimeter = grams_per_cubic_centimeter_t<float>;
using pounds_per_cubic_foot = pounds_per_cubic_foot_t<float>;
/// @}

/// @brief Type aliases for frequency quantities with float type
/// @{
using hertz = hertz_t<float>;
using kilohertz = kilohertz_t<float>;
using megahertz = megahertz_t<float>;
using gigahertz = gigahertz_t<float>;
/// @}

/// @brief Type aliases for angular velocity quantities with float type
/// @{
using radians_per_second = radians_per_second_t<float>;
using degrees_per_second = degrees_per_second_t<float>;
using revolutions_per_minute = revolutions_per_minute_t<float>;
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
/// @}

/// @brief Type aliases for dynamic viscosity quantities with float type
/// @{
using pascal_seconds = pascal_seconds_t<float>;
using poise = poise_t<float>;
/// @}

/// @brief Type aliases for kinematic viscosity quantities with float type
/// @{
using square_meters_per_second = square_meters_per_second_t<float>;
using stokes = stokes_t<float>;
/// @}

/// @brief Type aliases for surface tension quantities with float type
/// @{
using newtons_per_meter = newtons_per_meter_t<float>;
using dynes_per_centimeter = dynes_per_centimeter_t<float>;
/// @}

/// @brief Type aliases for specific heat capacity quantities with float type
/// @{
using joules_per_kilogram_kelvin = joules_per_kilogram_kelvin_t<float>;
using calories_per_gram_celsius = calories_per_gram_celsius_t<float>;
/// @}

/// @brief Type aliases for thermal conductivity quantities with float type
/// @{
using watts_per_meter_kelvin = watts_per_meter_kelvin_t<float>;
/// @}

/// @brief Type aliases for electric field strength quantities with float type
/// @{
using volts_per_meter = volts_per_meter_t<float>;
/// @}

/// @brief Type aliases for permittivity quantities with float type
/// @{
using farads_per_meter = farads_per_meter_t<float>;
/// @}

/// @brief Type aliases for permeability quantities with float type
/// @{
using henries_per_meter = henries_per_meter_t<float>;
/// @}

/// @brief Type aliases for absorbed dose quantities with float type
/// @{
using grays = grays_t<float>;
using rads = rads_t<float>;
/// @}

/// @brief Type aliases for equivalent dose quantities with float type
/// @{
using sieverts = sieverts_t<float>;
using rems = rems_t<float>;
/// @}

/// @brief Type aliases for catalytic activity quantities with float type
/// @{
using katal = katal_t<float>;
using enzyme_unit = enzyme_unit_t<float>;
/// @}

/// @brief Type aliases for concentration quantities with float type
/// @{
using moles_per_cubic_meter = moles_per_cubic_meter_t<float>;
using moles_per_liter = moles_per_liter_t<float>;
/// @}

/// @brief Type aliases for molality quantities with float type
/// @{
using moles_per_kilogram = moles_per_kilogram_t<float>;
/// @}

/// @brief Type aliases for molar mass quantities with float type
/// @{
using kilograms_per_mole = kilograms_per_mole_t<float>;
using grams_per_mole = grams_per_mole_t<float>;
/// @}

/// @brief Type aliases for luminous flux quantities with float type
/// @{
using lumens = lumens_t<float>;
/// @}

/// @brief Type aliases for illuminance quantities with float type
/// @{
using lux = lux_t<float>;
using foot_candles = foot_candles_t<float>;
/// @}

/// @brief Type aliases for luminous energy quantities with float type
/// @{
using lumen_seconds = lumen_seconds_t<float>;
/// @}

/// @brief Type aliases for luminous exposure quantities with float type
/// @{
using lux_seconds = lux_seconds_t<float>;
/// @}

/// @brief Type aliases for radioactivity quantities with float type
/// @{
using becquerels = becquerels_t<float>;
using curies = curies_t<float>;
/// @}

/// @brief Type aliases for flow rate quantities with float type
/// @{
using cubic_meters_per_second = cubic_meters_per_second_t<float>;
using liters_per_second = liters_per_second_t<float>;
using gallons_per_minute = gallons_per_minute_t<float>;
/// @}

// NOLINTEND
} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_TYPES_HPP