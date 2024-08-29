/**
 * @file dimension_types.hpp
 * @brief Defines dimension types and operations for physical quantities.
 *
 * This file provides a comprehensive set of dimension types used in physical calculations,
 * It includes base dimensions (such as length, time, mass), derived dimensions
 * (like velocity, energy), and utility types for dimension arithmetic. These types form
 * the foundation for type-safe physical quantity calculations in the squint library.
 */

#ifndef SQUINT_QUANTITY_DIMENSION_TYPES_HPP
#define SQUINT_QUANTITY_DIMENSION_TYPES_HPP

#include "squint/core/concepts.hpp"
#include "squint/quantity/dimension.hpp"

#include <concepts>
#include <ratio>

namespace squint {

/**
 * @brief Utility types for performing arithmetic operations on dimensions.
 */

/**
 * @brief Multiply two dimensions.
 * @tparam U1 The first dimension.
 * @tparam U2 The second dimension.
 */
template <dimensional U1, dimensional U2> using dim_mult_t = typename dim_mult<U1, U2>::type;

/**
 * @brief Divide two dimensions.
 * @tparam U1 The numerator dimension.
 * @tparam U2 The denominator dimension.
 */
template <dimensional U1, dimensional U2> using dim_div_t = typename dim_div<U1, U2>::type;

/**
 * @brief Raise a dimension to an integer power.
 * @tparam U The dimension to be raised.
 * @tparam N The power to raise the dimension to.
 */
template <dimensional U, std::integral auto const N> using dim_pow_t = typename dim_pow<U, N>::type;

/**
 * @brief Take the Nth root of a dimension.
 * @tparam U The dimension to take the root of.
 * @tparam N The root to take.
 */
template <dimensional U, std::integral auto const N> using dim_root_t = typename dim_root<U, N>::type;

/**
 * @brief Invert a dimension (raise to power -1).
 * @tparam U The dimension to invert.
 */
template <dimensional U>
using dim_inv_t = dim_div_t<
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>,
    U>;

/**
 * @brief Namespace containing common dimension definitions.
 *
 * This namespace provides a comprehensive set of dimension types used in physical calculations.
 * It includes both base dimensions (corresponding to SI base units) and derived dimensions.
 */
namespace dimensions {

/**
 * @brief Base dimensions corresponding to SI base units.
 *
 * These dimensions form the foundation of the dimensional system and correspond
 * to the seven SI base units.
 */

/** @brief Dimensionless quantity. */
using unity =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

/** @brief Length dimension. */
using L =
    dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

/** @brief Time dimension. */
using T =
    dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

/** @brief Mass dimension. */
using M =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

/** @brief Temperature dimension. */
using K =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

/** @brief Electric current dimension. */
using I =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>>;

/** @brief Amount of substance dimension. */
using N =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>>;

/** @brief Luminous intensity dimension. */
using J =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>>;

/**
 * @brief Derived dimensions based on SI base dimensions.
 *
 * These dimensions are derived from the seven SI base dimensions.
 */

/** @brief Velocity dimension (length/time). */
using velocity_dim = dim_div_t<L, T>;

/** @brief Acceleration dimension (velocity/time). */
using acceleration_dim = dim_div_t<velocity_dim, T>;

/** @brief Force dimension (mass * acceleration). */
using force_dim = dim_mult_t<M, acceleration_dim>;

/** @brief Energy dimension (force * length). */
using energy_dim = dim_mult_t<force_dim, L>;

/** @brief Power dimension (energy/time). */
using power_dim = dim_div_t<energy_dim, T>;

/** @brief Pressure dimension (force/area). */
using pressure_dim = dim_div_t<force_dim, dim_pow_t<L, 2>>;

/** @brief Electric charge dimension (current * time). */
using charge_dim = dim_mult_t<I, T>;

/** @brief Area dimension (length^2). */
using area_dim = dim_pow_t<L, 2>;

/** @brief Volume dimension (length^3). */
using volume_dim = dim_pow_t<L, 3>;

/** @brief Density dimension (mass/volume). */
using density_dim = dim_div_t<M, volume_dim>;

/** @brief Frequency dimension (1/time). */
using frequency_dim = dim_inv_t<T>;

/** @brief Angle dimension (dimensionless). */
using angle_dim = unity;

/** @brief Angular velocity dimension (angle/time). */
using angular_velocity_dim = dim_div_t<angle_dim, T>;

/** @brief Angular acceleration dimension (angular velocity/time). */
using angular_acceleration_dim = dim_div_t<angular_velocity_dim, T>;

/** @brief Torque dimension (force * length). */
using torque_dim = dim_mult_t<force_dim, L>;

/** @brief Moment of inertia dimension (mass * area). */
using moment_of_inertia_dim = dim_mult_t<M, area_dim>;

/** @brief Momentum dimension (mass * velocity). */
using momentum_dim = dim_mult_t<M, velocity_dim>;

/** @brief Angular momentum dimension (moment of inertia * angular velocity). */
using angular_momentum_dim = dim_mult_t<moment_of_inertia_dim, angular_velocity_dim>;

/** @brief Voltage dimension (power/current). */
using voltage_dim = dim_div_t<power_dim, I>;

/** @brief Electric resistance dimension (voltage/current). */
using resistance_dim = dim_div_t<voltage_dim, I>;

/** @brief Electrical conductance dimension (1/resistance). */
using conductance_dim = dim_inv_t<resistance_dim>;

/** @brief Capacitance dimension (charge/voltage). */
using capacitance_dim = dim_div_t<charge_dim, voltage_dim>;

/** @brief Magnetic flux dimension (voltage * time). */
using magnetic_flux_dim = dim_mult_t<voltage_dim, T>;

/** @brief Magnetic flux density dimension (magnetic flux / area). */
using magnetic_flux_density_dim = dim_div_t<magnetic_flux_dim, area_dim>;

/** @brief Inductance dimension (magnetic flux / current). */
using inductance_dim = dim_div_t<magnetic_flux_dim, I>;

/** @brief Electric field strength dimension (voltage / length). */
using electric_field_strength_dim = dim_div_t<voltage_dim, L>;

/** @brief Magnetic field strength dimension (current / length). */
using magnetic_field_strength_dim = dim_div_t<I, L>;

/** @brief Permittivity dimension (capacitance / length). */
using permittivity_dim = dim_div_t<capacitance_dim, L>;

/** @brief Permeability dimension (inductance / length). */
using permeability_dim = dim_div_t<inductance_dim, L>;

/** @brief Specific energy dimension (energy/mass). */
using specific_energy_dim = dim_div_t<energy_dim, M>;

/** @brief Specific heat capacity dimension (energy/(mass * temperature)). */
using specific_heat_capacity_dim = dim_div_t<energy_dim, dim_mult_t<M, K>>;

/** @brief Thermal conductivity dimension (power/(length * temperature)). */
using thermal_conductivity_dim = dim_div_t<power_dim, dim_mult_t<L, K>>;

/** @brief Dynamic viscosity dimension (pressure * time). */
using dynamic_viscosity_dim = dim_mult_t<pressure_dim, T>;

/** @brief Kinematic viscosity dimension (area/time). */
using kinematic_viscosity_dim = dim_div_t<area_dim, T>;

/** @brief Surface tension dimension (force/length). */
using surface_tension_dim = dim_div_t<force_dim, L>;

/** @brief Strain dimension (dimensionless). */
using strain_dim = unity;

/** @brief Stress dimension (force/area). */
using stress_dim = pressure_dim;

/** @brief Young's modulus dimension (stress/strain). */
using youngs_modulus_dim = stress_dim;

/** @brief Poisson's ratio dimension (dimensionless). */
using poissons_ratio_dim = unity;

/** @brief Bulk modulus dimension (pressure). */
using bulk_modulus_dim = pressure_dim;

/** @brief Shear modulus dimension (pressure). */
using shear_modulus_dim = pressure_dim;

/** @brief Spring constant dimension (force/length). */
using spring_constant_dim = dim_div_t<force_dim, L>;

/** @brief Damping coefficient dimension (force * time / length). */
using damping_coefficient_dim = dim_mult_t<force_dim, dim_div_t<T, L>>;

/** @brief Impulse dimension (force * time). */
using impulse_dim = dim_mult_t<force_dim, T>;

/** @brief Specific impulse dimension (impulse/mass). */
using specific_impulse_dim = dim_div_t<impulse_dim, M>;

/** @brief Diffusivity dimension (area/time). */
using diffusivity_dim = dim_div_t<area_dim, T>;

/** @brief Thermal diffusivity dimension (area/time). */
using thermal_diffusivity_dim = diffusivity_dim;

/** @brief Heat flux dimension (power/area). */
using heat_flux_dim = dim_div_t<power_dim, area_dim>;

/** @brief Entropy dimension (energy/temperature). */
using entropy_dim = dim_div_t<energy_dim, K>;

/** @brief Specific entropy dimension (entropy/mass). */
using specific_entropy_dim = dim_div_t<entropy_dim, M>;

/** @brief Molar entropy dimension (entropy/amount of substance). */
using molar_entropy_dim = dim_div_t<entropy_dim, N>;

/** @brief Molar mass dimension (mass/amount of substance). */
using molar_mass_dim = dim_div_t<M, N>;

/** @brief Luminous flux dimension (luminous intensity * solid angle). */
using luminous_flux_dim = dim_mult_t<J, angle_dim>;

/** @brief Illuminance dimension (luminous flux / area). */
using illuminance_dim = dim_div_t<luminous_flux_dim, area_dim>;

/** @brief Luminous energy dimension (luminous flux * time). */
using luminous_energy_dim = dim_mult_t<luminous_flux_dim, T>;

/** @brief Luminous exposure dimension (illuminance * time). */
using luminous_exposure_dim = dim_mult_t<illuminance_dim, T>;

/** @brief Radioactivity dimension (1/time). */
using radioactivity_dim = frequency_dim;

/** @brief Absorbed dose dimension (energy/mass). */
using absorbed_dose_dim = specific_energy_dim;

/** @brief Equivalent dose dimension (energy/mass). */
using equivalent_dose_dim = specific_energy_dim;

/** @brief Catalytic activity dimension (amount of substance / time). */
using catalytic_activity_dim = dim_div_t<N, T>;

/** @brief Concentration dimension (amount of substance / volume). */
using concentration_dim = dim_div_t<N, volume_dim>;

/** @brief Molality dimension (amount of substance / mass). */
using molality_dim = dim_div_t<N, M>;

/** @brief Flow dimension (volume / time). */
using flow_dim = dim_div_t<volume_dim, T>;

} // namespace dimensions

} // namespace squint

#endif // SQUINT_QUANTITY_DIMENSION_TYPES_HPP