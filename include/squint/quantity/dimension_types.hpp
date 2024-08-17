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
template <dimensional U1, dimensional U2> using dim_mult_t = typename dim_mult<U1, U2>::type;

/// @brief Divide two dimensions.
template <dimensional U1, dimensional U2> using dim_div_t = typename dim_div<U1, U2>::type;

/// @brief Raise a dimension to an integer power.
template <dimensional U, std::integral auto const N> using dim_pow_t = typename dim_pow<U, N>::type;

/// @brief Take the Nth root of a dimension.
template <dimensional U, std::integral auto const N> using dim_root_t = typename dim_root<U, N>::type;

/// @brief Invert a dimension (raise to power -1).
template <dimensional U>
using dim_inv_t = dim_div_t<
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>,
    U>;

/** @} */ // end of dimension_arithmetic group

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

using unity =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using L =
    dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using T =
    dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using M =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using K =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using I =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>>;
using N =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>>;
using J =
    dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>>;

/** @} */ // end of base_dimensions group

/**
 * @defgroup derived_dimensions Derived Dimensions
 * @brief Derived dimensions based on SI base dimensions.
 *
 * These dimensions are derived from the seven SI base dimensions.
 * @{
 */

using velocity_dim = dim_div_t<L, T>;
using acceleration_dim = dim_div_t<velocity_dim, T>;
using force_dim = dim_mult_t<M, acceleration_dim>;
using energy_dim = dim_mult_t<M, dim_pow_t<velocity_dim, 2>>;
using power_dim = dim_div_t<energy_dim, T>;
using pressure_dim = dim_div_t<force_dim, dim_pow_t<L, 2>>;
using charge_dim = dim_mult_t<I, T>;
using area_dim = dim_pow_t<L, 2>;
using volume_dim = dim_pow_t<L, 3>;
using density_dim = dim_div_t<M, dim_pow_t<L, 3>>;
using frequency_dim = dim_inv_t<T>;
using angle_dim = unity;
using angular_velocity_dim = dim_div_t<angle_dim, T>;
using angular_acceleration_dim = dim_div_t<angular_velocity_dim, T>;
using torque_dim = dim_mult_t<force_dim, L>;
using moment_of_inertia_dim = dim_mult_t<M, area_dim>;
using momentum_dim = dim_mult_t<M, velocity_dim>;
using voltage_dim = dim_div_t<power_dim, I>;
using inductance_dim = dim_div_t<dim_div_t<voltage_dim, I>, T>;
using capacitance_dim = dim_div_t<charge_dim, voltage_dim>;
using flow_dim = dim_div_t<volume_dim, T>;
using viscosity_dim = dim_mult_t<pressure_dim, T>;

/** @} */ // end of derived_dimensions group

} // namespace dimensions

} // namespace squint

#endif // SQUINT_QUANTITY_DIMENSION_TYPES_HPP