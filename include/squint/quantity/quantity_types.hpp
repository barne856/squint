/**
 * @file quantity_types.hpp
 * @brief Defines type aliases for various quantity specializations.
 *
 * This file provides convenient type aliases for quantities with different
 * error checking policies and for constant quantities.
 */

#ifndef SQUINT_QUANTITY_QUANTITY_TYPES_HPP
#define SQUINT_QUANTITY_QUANTITY_TYPES_HPP

#include "squint/core/concepts.hpp"
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
 * @defgroup base_quantities Base Quantities
 * @brief Base base_quantities corresponding to SI.
 * @{
 */

/// @brief Type alias for a dimensionless quantity.
template <typename T> using pure_t = unchecked_quantity_t<T, dimensions::unity>;
/// @brief Type alias for a length quantity.
template <typename T> using length_t = unchecked_quantity_t<T, dimensions::L>;
/// @brief Type alias for a time quantity.
template <typename T> using time_t = unchecked_quantity_t<T, dimensions::T>;
/// @brief Type alias for a mass quantity.
template <typename T> using mass_t = unchecked_quantity_t<T, dimensions::M>;
/// @brief Type alias for a temperature quantity.
template <typename T> using temperature_t = unchecked_quantity_t<T, dimensions::K>;
/// @brief Type alias for an electric current quantity.
template <typename T> using current_t = unchecked_quantity_t<T, dimensions::I>;
/// @brief Type alias for an amount of substance quantity.
template <typename T> using amount_t = unchecked_quantity_t<T, dimensions::N>;
/// @brief Type alias for a luminous intensity quantity.
template <typename T> using intensity_t = unchecked_quantity_t<T, dimensions::J>;

/** @} */ // end of base_quantities group

/**
 * @defgroup derived_quantities Derived Quantities
 * @brief Derived quantities based on SI base quantities.
 *
 * These quantities are derived from the seven SI base quantities.
 * @{
 */

/// @brief Type alias for a velocity quantity.
template <typename T> using velocity_t = unchecked_quantity_t<T, dimensions::velocity_dim>;
/// @brief Type alias for an acceleration quantity.
template <typename T> using acceleration_t = unchecked_quantity_t<T, dimensions::acceleration_dim>;
/// @brief Type alias for a force quantity.
template <typename T> using force_t = unchecked_quantity_t<T, dimensions::force_dim>;
/// @brief Type alias for an energy quantity.
template <typename T> using energy_t = unchecked_quantity_t<T, dimensions::energy_dim>;
/// @brief Type alias for a power quantity.
template <typename T> using power_t = unchecked_quantity_t<T, dimensions::power_dim>;
/// @brief Type alias for a pressure quantity.
template <typename T> using pressure_t = unchecked_quantity_t<T, dimensions::pressure_dim>;
/// @brief Type alias for an electric charge quantity.
template <typename T> using charge_t = unchecked_quantity_t<T, dimensions::charge_dim>;
/// @brief Type alias for an area quantity.
template <typename T> using area_t = unchecked_quantity_t<T, dimensions::area_dim>;
/// @brief Type alias for a volume quantity.
template <typename T> using volume_t = unchecked_quantity_t<T, dimensions::volume_dim>;
/// @brief Type alias for a density quantity.
template <typename T> using density_t = unchecked_quantity_t<T, dimensions::density_dim>;
/// @brief Type alias for a frequency quantity.
template <typename T> using frequency_t = unchecked_quantity_t<T, dimensions::frequency_dim>;
/// @brief Type alias for an angle quantity.
template <typename T> using angle_t = unchecked_quantity_t<T, dimensions::angle_dim>;
/// @brief Type alias for an angular velocity quantity.
template <typename T> using angular_velocity_t = unchecked_quantity_t<T, dimensions::angular_velocity_dim>;
/// @brief Type alias for an angular acceleration quantity.
template <typename T> using angular_acceleration_t = unchecked_quantity_t<T, dimensions::angular_acceleration_dim>;
/// @brief Type alias for a torque quantity.
template <typename T> using torque_t = unchecked_quantity_t<T, dimensions::torque_dim>;
/// @brief Type alias for a moment of inertia quantity.
template <typename T> using moment_of_inertia_t = unchecked_quantity_t<T, dimensions::moment_of_inertia_dim>;
/// @brief Type alias for a linear momentum quantity.
template <typename T> using momentum_t = unchecked_quantity_t<T, dimensions::momentum_dim>;
/// @brief Type alias for a voltage quantity.
template <typename T> using voltage_t = unchecked_quantity_t<T, dimensions::voltage_dim>;
/// @brief Type alias for an inductance quantity.
template <typename T> using inductance_t = unchecked_quantity_t<T, dimensions::inductance_dim>;
/// @brief Type alias for a capacitance quantity.
template <typename T> using capacitance_t = unchecked_quantity_t<T, dimensions::capacitance_dim>;
/// @ brief Type alias for a flow quantity.
template <typename T> using flow_t = unchecked_quantity_t<T, dimensions::flow_dim>;
/// @brief Type alias for a viscosity quantity.
template <typename T> using viscosity_t = unchecked_quantity_t<T, dimensions::viscosity_dim>;

/** @} */ // end of derived_quantities group

/**
 * @defgroup float_base_quantities
 * @brief Base quantities based on SI base quantities with single precision floating-point types.
 *
 * These quantities are the base seven SI base quantities and use floating-point types.
 * @{
 */

/// @brief Type alias for a dimensionless quantity with a float type.
using pure = pure_t<float>;
/// @brief Type alias for a length quantity with a float type.
using length = length_t<float>;
/// @brief Type alias for a time quantity with a float type.
using time = time_t<float>;
/// @brief Type alias for a mass quantity with a float type.
using mass = mass_t<float>;
/// @brief Type alias for a temperature quantity with a float type.
using temperature = temperature_t<float>;
/// @brief Type alias for an electric current quantity with a float type.
using current = current_t<float>;
/// @brief Type alias for an amount of substance quantity with a float type.
using amount = amount_t<float>;
/// @brief Type alias for a luminous intensity quantity with a float type.
using intensity = intensity_t<float>;

/** @} */ // end of float_base_quantities group

/**
 * @defgroup float_derived_quantities
 * @brief Derived quantities based on SI base quantities with single precision floating-point types.
 *
 * These quantities are derived from the seven SI base quantities and use floating-point types.
 * @{
 */

/// @brief Type alias for a velocity quantity with a float type.
using velocity = velocity_t<float>;
/// @brief Type alias for an acceleration quantity with a float type.
using acceleration = acceleration_t<float>;
/// @brief Type alias for a force quantity with a float type.
using force = force_t<float>;
/// @brief Type alias for an energy quantity with a float type.
using energy = energy_t<float>;
/// @brief Type alias for a power quantity with a float type.
using power = power_t<float>;
/// @brief Type alias for a pressure quantity with a float type.
using pressure = pressure_t<float>;
/// @brief Type alias for an electric charge quantity with a float type.
using charge = charge_t<float>;
/// @brief Type alias for an area quantity with a float type.
using area = area_t<float>;
/// @brief Type alias for a volume quantity with a float type.
using volume = volume_t<float>;
/// @brief Type alias for a density quantity with a float type.
using density = density_t<float>;
/// @brief Type alias for a frequency quantity with a float type.
using frequency = frequency_t<float>;
/// @brief Type alias for an angle quantity with a float type.
using angle = angle_t<float>;
/// @brief Type alias for an angular velocity quantity with a float type.
using angular_velocity = angular_velocity_t<float>;
/// @brief Type alias for an angular acceleration quantity with a float type.
using angular_acceleration = angular_acceleration_t<float>;
/// @brief Type alias for a torque quantity with a float type.
using torque = torque_t<float>;
/// @brief Type alias for a moment of inertia quantity with a float type.
using moment_of_inertia = moment_of_inertia_t<float>;
/// @brief Type alias for a linear momentum quantity with a float type.
using momentum = momentum_t<float>;
/// @brief Type alias for a voltage quantity with a float type.
using voltage = voltage_t<float>;
/// @brief Type alias for an inductance quantity with a float type.
using inductance = inductance_t<float>;
/// @brief Type alias for a capacitance quantity with a float type.
using capacitance = capacitance_t<float>;
/// @brief Type alias for a flow quantity with a float type.
using flow = flow_t<float>;
/// @brief Type alias for a viscosity quantity with a float type.
using viscosity = viscosity_t<float>;

/** @} */ // end of float_derived_quantities group

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_TYPES_HPP