/**
 * @file dimension_types.hpp
 * @brief Defines dimension types and operations for physical quantities.
 *
 * This file provides a comprehensive set of dimension types used in physical calculations.
 * It includes base dimensions (such as length, time, mass), derived dimensions (like velocity, energy),
 * and utility types for dimension arithmetic. These types form the foundation for
 * type-safe physical quantity calculations in the squint library.
 */

#ifndef SQUINT_QUANTITY_DIMENSION_TYPES_HPP
#define SQUINT_QUANTITY_DIMENSION_TYPES_HPP

#include "squint/quantity/dimension.hpp"

namespace squint {

/**
 * @defgroup dimension_arithmetic Dimension Arithmetic
 * @brief Utility types for performing arithmetic operations on dimensions.
 * @{
 */

/// @brief Multiply two dimensions.
template <dimensional U1, dimensional U2> using mult_t = typename dim_mult<U1, U2>::type;

/// @brief Divide two dimensions.
template <dimensional U1, dimensional U2> using div_t = typename dim_div<U1, U2>::type;

/// @brief Raise a dimension to an integer power.
template <dimensional U, std::integral auto const N> using pow_t = typename dim_pow<U, N>::type;

/// @brief Take the Nth root of a dimension.
template <dimensional U, std::integral auto const N> using root_t = typename dim_root<U, N>::type;

/// @brief Invert a dimension (raise to power -1).
template <dimensional U>
using inv_t = div_t<
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>,
    U>;

/** @} */  // end of dimension_arithmetic group

/**
 * @brief Namespace containing common dimension definitions.
 *
 * This namespace provides a comprehensive set of dimension types used in physical calculations.
 * It includes both base dimensions (corresponding to SI base units) and derived dimensions.
 */
namespace dimensions {

/**
 * @defgroup base_dimensions Base Dimensions
 * @brief Base dimensions corresponding to SI base units.
 *
 * These dimensions form the foundation of the dimensional system and correspond
 * to the seven SI base units.
 * @{
 */

using dimensionless =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using length =
    dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using time =
    dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using mass =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using temperature =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using current =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>>;
using amount_of_substance =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>>;
using luminous_intensity =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>>;

/** @} */  // end of base_dimensions group

/**
 * @defgroup other_dimensionless Other Dimensionless Quantities
 * @brief Quantities that are ratios of quantities with the same dimension.
 *
 * These are quantities that result in a dimensionless value.
 * @{
 */

using angle = dimensionless;
using solid_angle = dimensionless;
using strain = dimensionless;
using refractive_index = dimensionless;

/** @} */  // end of other_dimensionless group

/**
 * @defgroup derived_dimensions Derived Dimensions
 * @brief Dimensions derived from base dimensions through various combinations of multiplication and division operations.
 * @{
 */

/// @brief Velocity (length/time)
using velocity = div_t<length, time>;

/// @brief Acceleration (velocity/time)
using acceleration = div_t<velocity, time>;

/// @brief Area (length^2)
using area = mult_t<length, length>;

/// @brief Volume (length^3)
using volume = mult_t<area, length>;

/// @brief Density (mass/volume)
using density = div_t<mass, volume>;

/// @brief Force (mass * acceleration)
using force = mult_t<mass, acceleration>;

/// @brief Force density (force/volume)
using force_density = div_t<force, volume>;

/// @brief Pressure (force/area)
using pressure = div_t<force, area>;

/// @brief Dynamic viscosity (pressure * time)
using dynamic_viscosity = mult_t<pressure, time>;

/// @brief Kinematic viscosity (area/time)
using kinematic_viscosity = div_t<area, time>;

/// @brief Flow (volume/time)
using flow = div_t<volume, time>;

/// @brief Energy (force * length)
using energy = mult_t<force, length>;

/// @brief Power (energy/time)
using power = div_t<energy, time>;

/// @brief Electric charge (current * time)
using charge = mult_t<current, time>;

/// @brief Voltage (energy/charge)
using voltage = div_t<energy, charge>;

/// @brief Capacitance (charge/voltage)
using capacitance = div_t<charge, voltage>;

/// @brief Resistance (voltage/current)
using resistance = div_t<voltage, current>;

/// @brief Conductance (1/resistance)
using conductance = inv_t<resistance>;

/// @brief Magnetic flux (voltage * time)
using magnetic_flux = mult_t<voltage, time>;

/// @brief Magnetic flux density (magnetic_flux/area)
using magnetic_flux_density = div_t<magnetic_flux, area>;

/// @brief Inductance (magnetic_flux/current)
using inductance = div_t<magnetic_flux, current>;

/// @brief Frequency (1/time)
using frequency = inv_t<time>;

/// @brief Angular velocity (angle/time)
using angular_velocity = div_t<angle, time>;

/// @brief Momentum (mass * velocity)
using momentum = mult_t<mass, velocity>;

/// @brief Angular momentum (momentum * length)
using angular_momentum = mult_t<momentum, length>;

/// @brief Torque (force * length)
using torque = mult_t<force, length>;

/// @brief Surface tension (force/length)
using surface_tension = div_t<force, length>;

/// @brief Heat capacity (energy/temperature)
using heat_capacity = div_t<energy, temperature>;

/// @brief Specific heat capacity (heat_capacity/mass)
using specific_heat_capacity = div_t<heat_capacity, mass>;

/// @brief Thermal conductivity (power / (length * temperature))
using thermal_conductivity = div_t<power, mult_t<length, temperature>>;

/// @brief Electric field strength (force/charge)
using electric_field_strength = div_t<force, charge>;

/// @brief Electric displacement (charge/area)
using electric_displacement = div_t<charge, area>;

/// @brief Permittivity (capacitance/length)
using permittivity = div_t<capacitance, length>;

/// @brief Permeability (inductance/length)
using permeability = mult_t<inductance, inv_t<length>>;

/// @brief Molar energy (energy/amount_of_substance)
using molar_energy = div_t<energy, amount_of_substance>;

/// @brief Molar entropy (molar_energy/temperature)
using molar_entropy = div_t<molar_energy, temperature>;

/// @brief Exposure (charge/mass)
using exposure = div_t<charge, mass>;

/// @brief Dose equivalent (energy/mass)
using dose_equivalent = div_t<energy, mass>;

/// @brief Catalytic activity (amount_of_substance/time)
using catalytic_activity = div_t<amount_of_substance, time>;

/// @brief Luminance (luminous_intensity/area)
using luminance = div_t<luminous_intensity, area>;

/// @brief Magnetic field strength (current/length)
using magnetic_field_strength = div_t<current, length>;

/// @brief Molarity (amount_of_substance/volume)
using molarity = div_t<amount_of_substance, volume>;

/// @brief Molar mass (mass/amount_of_substance)
using molar_mass = div_t<mass, amount_of_substance>;

/// @brief Impulse (force * time)
using impulse = mult_t<force, time>;

/// @brief Wave number (1/length)
using wave_number = inv_t<length>;

/// @brief Specific volume (volume/mass)
using specific_volume = div_t<volume, mass>;

/// @brief Radiant intensity (power/solid_angle)
using radiant_intensity = div_t<power, solid_angle>;

/// @brief Radiance (radiant_intensity/area)
using radiance = div_t<radiant_intensity, area>;

/// @brief Irradiance (power/area)
using irradiance = div_t<power, area>;

/// @brief Thermal resistance (temperature/power)
using thermal_resistance = div_t<temperature, power>;

/** @} */  // end of derived_dimensions group

} // namespace dimensions

} // namespace squint

#endif // SQUINT_QUANTITY_DIMENSION_TYPES_HPP