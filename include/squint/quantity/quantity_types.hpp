/**
 * @file quantity_types.hpp
 * @brief Defines type aliases for various quantity specializations.
 *
 * This file provides convenient type aliases for quantities with different
 * error checking policies and for constant quantities. It covers a wide range
 * of physical quantities based on the dimensions defined in dimension_types.hpp.
 */

#ifndef SQUINT_QUANTITY_QUANTITY_TYPES_HPP
#define SQUINT_QUANTITY_QUANTITY_TYPES_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/quantity.hpp"

namespace squint {

/**
 * @brief Type alias for quantities with error checking enabled.
 *
 * @tparam T The underlying arithmetic type of the quantity.
 * @tparam D The dimension type of the quantity.
 */
template <typename T, dimensional D> using checked_quantity_t = quantity<T, D, error_checking::enabled>;

/**
 * @brief Type alias for quantities with error checking disabled.
 *
 * @tparam T The underlying arithmetic type of the quantity.
 * @tparam D The dimension type of the quantity.
 */
template <typename T, dimensional D> using unchecked_quantity_t = quantity<T, D, error_checking::disabled>;

/**
 * @brief Template alias for constant quantities.
 *
 * This alias creates an unchecked quantity with a floating-point type.
 * It's typically used for representing physical constants.
 *
 * @tparam T The underlying floating-point type of the quantity.
 * @tparam Dimension The dimension type of the quantity.
 */
template <floating_point T, typename Dimension> using constant_quantity_t = unchecked_quantity_t<T, Dimension>;

/**
 * @brief Base quantities corresponding to SI base units.
 */

/** @brief Type alias for a dimensionless quantity. */
template <typename T> using pure_t = unchecked_quantity_t<T, dimensions::unity>;
/** @brief Type alias for a length quantity. */
template <typename T> using length_t = unchecked_quantity_t<T, dimensions::L>;
/** @brief Type alias for a duration quantity. */
template <typename T> using duration_t = unchecked_quantity_t<T, dimensions::T>;
/** @brief Type alias for a mass quantity. */
template <typename T> using mass_t = unchecked_quantity_t<T, dimensions::M>;
/** @brief Type alias for a temperature quantity. */
template <typename T> using temperature_t = unchecked_quantity_t<T, dimensions::K>;
/** @brief Type alias for an electric current quantity. */
template <typename T> using current_t = unchecked_quantity_t<T, dimensions::I>;
/** @brief Type alias for an amount of substance quantity. */
template <typename T> using amount_t = unchecked_quantity_t<T, dimensions::N>;
/** @brief Type alias for a luminous intensity quantity. */
template <typename T> using luminous_intensity_t = unchecked_quantity_t<T, dimensions::J>;

/**
 * @brief Derived quantities based on SI base quantities.
 */

/** @brief Type alias for a velocity quantity. */
template <typename T> using velocity_t = unchecked_quantity_t<T, dimensions::velocity_dim>;
/** @brief Type alias for an acceleration quantity. */
template <typename T> using acceleration_t = unchecked_quantity_t<T, dimensions::acceleration_dim>;
/** @brief Type alias for a force quantity. */
template <typename T> using force_t = unchecked_quantity_t<T, dimensions::force_dim>;
/** @brief Type alias for an energy quantity. */
template <typename T> using energy_t = unchecked_quantity_t<T, dimensions::energy_dim>;
/** @brief Type alias for a power quantity. */
template <typename T> using power_t = unchecked_quantity_t<T, dimensions::power_dim>;
/** @brief Type alias for a pressure quantity. */
template <typename T> using pressure_t = unchecked_quantity_t<T, dimensions::pressure_dim>;
/** @brief Type alias for an electric charge quantity. */
template <typename T> using charge_t = unchecked_quantity_t<T, dimensions::charge_dim>;
/** @brief Type alias for an area quantity. */
template <typename T> using area_t = unchecked_quantity_t<T, dimensions::area_dim>;
/** @brief Type alias for a volume quantity. */
template <typename T> using volume_t = unchecked_quantity_t<T, dimensions::volume_dim>;
/** @brief Type alias for a density quantity. */
template <typename T> using density_t = unchecked_quantity_t<T, dimensions::density_dim>;
/** @brief Type alias for a frequency quantity. */
template <typename T> using frequency_t = unchecked_quantity_t<T, dimensions::frequency_dim>;
/** @brief Type alias for an angle quantity. */
template <typename T> using angle_t = unchecked_quantity_t<T, dimensions::angle_dim>;
/** @brief Type alias for an angular velocity quantity. */
template <typename T> using angular_velocity_t = unchecked_quantity_t<T, dimensions::angular_velocity_dim>;
/** @brief Type alias for an angular acceleration quantity. */
template <typename T> using angular_acceleration_t = unchecked_quantity_t<T, dimensions::angular_acceleration_dim>;
/** @brief Type alias for a torque quantity. */
template <typename T> using torque_t = unchecked_quantity_t<T, dimensions::torque_dim>;
/** @brief Type alias for a moment of inertia quantity. */
template <typename T> using moment_of_inertia_t = unchecked_quantity_t<T, dimensions::moment_of_inertia_dim>;
/** @brief Type alias for a linear momentum quantity. */
template <typename T> using momentum_t = unchecked_quantity_t<T, dimensions::momentum_dim>;
/** @brief Type alias for an angular momentum quantity. */
template <typename T> using angular_momentum_t = unchecked_quantity_t<T, dimensions::angular_momentum_dim>;
/** @brief Type alias for a voltage quantity. */
template <typename T> using voltage_t = unchecked_quantity_t<T, dimensions::voltage_dim>;
/** @brief Type alias for an electric resistance quantity. */
template <typename T> using resistance_t = unchecked_quantity_t<T, dimensions::resistance_dim>;
/** @brief Type alias for an electrical conductance quantity. */
template <typename T> using conductance_t = unchecked_quantity_t<T, dimensions::conductance_dim>;
/** @brief Type alias for a capacitance quantity. */
template <typename T> using capacitance_t = unchecked_quantity_t<T, dimensions::capacitance_dim>;
/** @brief Type alias for a magnetic flux quantity. */
template <typename T> using magnetic_flux_t = unchecked_quantity_t<T, dimensions::magnetic_flux_dim>;
/** @brief Type alias for a magnetic flux density quantity. */
template <typename T> using magnetic_flux_density_t = unchecked_quantity_t<T, dimensions::magnetic_flux_density_dim>;
/** @brief Type alias for an inductance quantity. */
template <typename T> using inductance_t = unchecked_quantity_t<T, dimensions::inductance_dim>;
/** @brief Type alias for an electric field strength quantity. */
template <typename T>
using electric_field_strength_t = unchecked_quantity_t<T, dimensions::electric_field_strength_dim>;
/** @brief Type alias for a magnetic field strength quantity. */
template <typename T>
using magnetic_field_strength_t = unchecked_quantity_t<T, dimensions::magnetic_field_strength_dim>;
/** @brief Type alias for a permittivity quantity. */
template <typename T> using permittivity_t = unchecked_quantity_t<T, dimensions::permittivity_dim>;
/** @brief Type alias for a permeability quantity. */
template <typename T> using permeability_t = unchecked_quantity_t<T, dimensions::permeability_dim>;
/** @brief Type alias for a specific energy quantity. */
template <typename T> using specific_energy_t = unchecked_quantity_t<T, dimensions::specific_energy_dim>;
/** @brief Type alias for a specific heat capacity quantity. */
template <typename T> using specific_heat_capacity_t = unchecked_quantity_t<T, dimensions::specific_heat_capacity_dim>;
/** @brief Type alias for a thermal conductivity quantity. */
template <typename T> using thermal_conductivity_t = unchecked_quantity_t<T, dimensions::thermal_conductivity_dim>;
/** @brief Type alias for a dynamic viscosity quantity. */
template <typename T> using dynamic_viscosity_t = unchecked_quantity_t<T, dimensions::dynamic_viscosity_dim>;
/** @brief Type alias for a kinematic viscosity quantity. */
template <typename T> using kinematic_viscosity_t = unchecked_quantity_t<T, dimensions::kinematic_viscosity_dim>;
/** @brief Type alias for a surface tension quantity. */
template <typename T> using surface_tension_t = unchecked_quantity_t<T, dimensions::surface_tension_dim>;
/** @brief Type alias for a strain quantity. */
template <typename T> using strain_t = unchecked_quantity_t<T, dimensions::strain_dim>;
/** @brief Type alias for a stress quantity. */
template <typename T> using stress_t = unchecked_quantity_t<T, dimensions::stress_dim>;
/** @brief Type alias for a Young's modulus quantity. */
template <typename T> using youngs_modulus_t = unchecked_quantity_t<T, dimensions::youngs_modulus_dim>;
/** @brief Type alias for a Poisson's ratio quantity. */
template <typename T> using poissons_ratio_t = unchecked_quantity_t<T, dimensions::poissons_ratio_dim>;
/** @brief Type alias for a bulk modulus quantity. */
template <typename T> using bulk_modulus_t = unchecked_quantity_t<T, dimensions::bulk_modulus_dim>;
/** @brief Type alias for a shear modulus quantity. */
template <typename T> using shear_modulus_t = unchecked_quantity_t<T, dimensions::shear_modulus_dim>;
/** @brief Type alias for a spring constant quantity. */
template <typename T> using spring_constant_t = unchecked_quantity_t<T, dimensions::spring_constant_dim>;
/** @brief Type alias for a damping coefficient quantity. */
template <typename T> using damping_coefficient_t = unchecked_quantity_t<T, dimensions::damping_coefficient_dim>;
/** @brief Type alias for an impulse quantity. */
template <typename T> using impulse_t = unchecked_quantity_t<T, dimensions::impulse_dim>;
/** @brief Type alias for a specific impulse quantity. */
template <typename T> using specific_impulse_t = unchecked_quantity_t<T, dimensions::specific_impulse_dim>;
/** @brief Type alias for a diffusivity quantity. */
template <typename T> using diffusivity_t = unchecked_quantity_t<T, dimensions::diffusivity_dim>;
/** @brief Type alias for a thermal diffusivity quantity. */
template <typename T> using thermal_diffusivity_t = unchecked_quantity_t<T, dimensions::thermal_diffusivity_dim>;
/** @brief Type alias for a heat flux quantity. */
template <typename T> using heat_flux_t = unchecked_quantity_t<T, dimensions::heat_flux_dim>;
/** @brief Type alias for an entropy quantity. */
template <typename T> using entropy_t = unchecked_quantity_t<T, dimensions::entropy_dim>;
/** @brief Type alias for a specific entropy quantity. */
template <typename T> using specific_entropy_t = unchecked_quantity_t<T, dimensions::specific_entropy_dim>;
/** @brief Type alias for a molar entropy quantity. */
template <typename T> using molar_entropy_t = unchecked_quantity_t<T, dimensions::molar_entropy_dim>;
/** @brief Type alias for a molar mass quantity. */
template <typename T> using molar_mass_t = unchecked_quantity_t<T, dimensions::molar_mass_dim>;
/** @brief Type alias for a luminous flux quantity. */
template <typename T> using luminous_flux_t = unchecked_quantity_t<T, dimensions::luminous_flux_dim>;
/** @brief Type alias for an illuminance quantity. */
template <typename T> using illuminance_t = unchecked_quantity_t<T, dimensions::illuminance_dim>;
/** @brief Type alias for a luminous energy quantity. */
template <typename T> using luminous_energy_t = unchecked_quantity_t<T, dimensions::luminous_energy_dim>;
/** @brief Type alias for a luminous exposure quantity. */
template <typename T> using luminous_exposure_t = unchecked_quantity_t<T, dimensions::luminous_exposure_dim>;
/** @brief Type alias for a radioactivity quantity. */
template <typename T> using radioactivity_t = unchecked_quantity_t<T, dimensions::radioactivity_dim>;
/** @brief Type alias for an absorbed dose quantity. */
template <typename T> using absorbed_dose_t = unchecked_quantity_t<T, dimensions::absorbed_dose_dim>;
/** @brief Type alias for an equivalent dose quantity. */
template <typename T> using equivalent_dose_t = unchecked_quantity_t<T, dimensions::equivalent_dose_dim>;
/** @brief Type alias for a catalytic activity quantity. */
template <typename T> using catalytic_activity_t = unchecked_quantity_t<T, dimensions::catalytic_activity_dim>;
/** @brief Type alias for a concentration quantity. */
template <typename T> using concentration_t = unchecked_quantity_t<T, dimensions::concentration_dim>;
/** @brief Type alias for a molality quantity. */
template <typename T> using molality_t = unchecked_quantity_t<T, dimensions::molality_dim>;
/** @brief Type alias for a flow quantity */
template <typename T> using flow_t = unchecked_quantity_t<T, dimensions::flow_dim>;

/**
 * @brief Base quantities with single precision floating-point types.
 */

/** @brief Type alias for a dimensionless quantity with a float type. */
using pure = pure_t<float>;
/** @brief Type alias for a length quantity with a float type. */
using length = length_t<float>;
/** @brief Type alias for a duration quantity with a float type. */
using duration = duration_t<float>;
/** @brief Type alias for a mass quantity with a float type. */
using mass = mass_t<float>;
/** @brief Type alias for a temperature quantity with a float type. */
using temperature = temperature_t<float>;
/** @brief Type alias for an electric current quantity with a float type. */
using current = current_t<float>;
/** @brief Type alias for an amount of substance quantity with a float type. */
using amount = amount_t<float>;
/** @brief Type alias for a luminous intensity quantity with a float type. */
using luminous_intensity = luminous_intensity_t<float>;

/**
 * @brief Derived quantities with single precision floating-point types.
 */

/** @brief Type alias for a velocity quantity with a float type. */
using velocity = velocity_t<float>;
/** @brief Type alias for an acceleration quantity with a float type. */
using acceleration = acceleration_t<float>;
/** @brief Type alias for a force quantity with a float type. */
using force = force_t<float>;
/** @brief Type alias for an energy quantity with a float type. */
using energy = energy_t<float>;
/** @brief Type alias for a power quantity with a float type. */
using power = power_t<float>;
/** @brief Type alias for a pressure quantity with a float type. */
using pressure = pressure_t<float>;
/** @brief Type alias for an electric charge quantity with a float type. */
using charge = charge_t<float>;
/** @brief Type alias for an area quantity with a float type. */
using area = area_t<float>;
/** @brief Type alias for a volume quantity with a float type. */
using volume = volume_t<float>;
/** @brief Type alias for a density quantity with a float type. */
using density = density_t<float>;
/** @brief Type alias for a frequency quantity with a float type. */
using frequency = frequency_t<float>;
/** @brief Type alias for an angle quantity with a float type. */
using angle = angle_t<float>;
/** @brief Type alias for an angular velocity quantity with a float type. */
using angular_velocity = angular_velocity_t<float>;
/** @brief Type alias for an angular acceleration quantity with a float type. */
using angular_acceleration = angular_acceleration_t<float>;
/** @brief Type alias for a torque quantity with a float type. */
using torque = torque_t<float>;
/** @brief Type alias for a moment of inertia quantity with a float type. */
using moment_of_inertia = moment_of_inertia_t<float>;
/** @brief Type alias for a linear momentum quantity with a float type. */
using momentum = momentum_t<float>;
/** @brief Type alias for an angular momentum quantity with a float type. */
using angular_momentum = angular_momentum_t<float>;
/** @brief Type alias for a voltage quantity with a float type. */
using voltage = voltage_t<float>;
/** @brief Type alias for an electric resistance quantity with a float type. */
using resistance = resistance_t<float>;
/** @brief Type alias for an electrical conductance quantity with a float type. */
using conductance = conductance_t<float>;
/** @brief Type alias for a capacitance quantity with a float type. */
using capacitance = capacitance_t<float>;
/** @brief Type alias for a magnetic flux quantity with a float type. */
using magnetic_flux = magnetic_flux_t<float>;
/** @brief Type alias for a magnetic flux density quantity with a float type. */
using magnetic_flux_density = magnetic_flux_density_t<float>;
/** @brief Type alias for an inductance quantity with a float type. */
using inductance = inductance_t<float>;
/** @brief Type alias for an electric field strength quantity with a float type. */
using electric_field_strength = electric_field_strength_t<float>;
/** @brief Type alias for a magnetic field strength quantity with a float type. */
using magnetic_field_strength = magnetic_field_strength_t<float>;
/** @brief Type alias for a permittivity quantity with a float type. */
using permittivity = permittivity_t<float>;
/** @brief Type alias for a permeability quantity with a float type. */
using permeability = permeability_t<float>;
/** @brief Type alias for a specific energy quantity with a float type. */
using specific_energy = specific_energy_t<float>;
/** @brief Type alias for a specific heat capacity quantity with a float type. */
using specific_heat_capacity = specific_heat_capacity_t<float>;
/** @brief Type alias for a thermal conductivity quantity with a float type. */
using thermal_conductivity = thermal_conductivity_t<float>;
/** @brief Type alias for a dynamic viscosity quantity with a float type. */
using dynamic_viscosity = dynamic_viscosity_t<float>;
/** @brief Type alias for a kinematic viscosity quantity with a float type. */
using kinematic_viscosity = kinematic_viscosity_t<float>;
/** @brief Type alias for a surface tension quantity with a float type. */
using surface_tension = surface_tension_t<float>;
/** @brief Type alias for a strain quantity with a float type. */
using strain = strain_t<float>;
/** @brief Type alias for a stress quantity with a float type. */
using stress = stress_t<float>;
/** @brief Type alias for a Young's modulus quantity with a float type. */
using youngs_modulus = youngs_modulus_t<float>;
/** @brief Type alias for a Poisson's ratio quantity with a float type. */
using poissons_ratio = poissons_ratio_t<float>;
/** @brief Type alias for a bulk modulus quantity with a float type. */
using bulk_modulus = bulk_modulus_t<float>;
/** @brief Type alias for a shear modulus quantity with a float type. */
using shear_modulus = shear_modulus_t<float>;
/** @brief Type alias for a spring constant quantity with a float type. */
using spring_constant = spring_constant_t<float>;
/** @brief Type alias for a damping coefficient quantity with a float type. */
using damping_coefficient = damping_coefficient_t<float>;
/** @brief Type alias for an impulse quantity with a float type. */
using impulse = impulse_t<float>;
/** @brief Type alias for a specific impulse quantity with a float type. */
using specific_impulse = specific_impulse_t<float>;
/** @brief Type alias for a diffusivity quantity with a float type. */
using diffusivity = diffusivity_t<float>;
/** @brief Type alias for a thermal diffusivity quantity with a float type. */
using thermal_diffusivity = thermal_diffusivity_t<float>;
/** @brief Type alias for a heat flux quantity with a float type. */
using heat_flux = heat_flux_t<float>;
/** @brief Type alias for an entropy quantity with a float type. */
using entropy = entropy_t<float>;
/** @brief Type alias for a specific entropy quantity with a float type. */
using specific_entropy = specific_entropy_t<float>;
/** @brief Type alias for a molar entropy quantity with a float type. */
using molar_entropy = molar_entropy_t<float>;
/** @brief Type alias for a molar mass quantity with a float type. */
using molar_mass = molar_mass_t<float>;
/** @brief Type alias for a luminous flux quantity with a float type. */
using luminous_flux = luminous_flux_t<float>;
/** @brief Type alias for an illuminance quantity with a float type. */
using illuminance = illuminance_t<float>;
/** @brief Type alias for a luminous energy quantity with a float type. */
using luminous_energy = luminous_energy_t<float>;
/** @brief Type alias for a luminous exposure quantity with a float type. */
using luminous_exposure = luminous_exposure_t<float>;
/** @brief Type alias for a radioactivity quantity with a float type. */
using radioactivity = radioactivity_t<float>;
/** @brief Type alias for an absorbed dose quantity with a float type. */
using absorbed_dose = absorbed_dose_t<float>;
/** @brief Type alias for an equivalent dose quantity with a float type. */
using equivalent_dose = equivalent_dose_t<float>;
/** @brief Type alias for a catalytic activity quantity with a float type. */
using catalytic_activity = catalytic_activity_t<float>;
/** @brief Type alias for a concentration quantity with a float type. */
using concentration = concentration_t<float>;
/** @brief Type alias for a molality quantity with a float type. */
using molality = molality_t<float>;
/** @brief Type alias for a flow quantity with a float type. */
using flow = flow_t<float>;

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_TYPES_HPP