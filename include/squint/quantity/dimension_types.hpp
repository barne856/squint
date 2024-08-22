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

/** @brief Energy dimension (mass * velocity^2). */
using energy_dim = dim_mult_t<M, dim_pow_t<velocity_dim, 2>>;

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
using density_dim = dim_div_t<M, dim_pow_t<L, 3>>;

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

/** @brief Voltage dimension (power/current). */
using voltage_dim = dim_div_t<power_dim, I>;

/** @brief Inductance dimension (voltage * time / current). */
using inductance_dim = dim_div_t<dim_div_t<voltage_dim, I>, T>;

/** @brief Capacitance dimension (charge/voltage). */
using capacitance_dim = dim_div_t<charge_dim, voltage_dim>;

/** @brief Flow dimension (volume/time). */
using flow_dim = dim_div_t<volume_dim, T>;

/** @brief Viscosity dimension (pressure * time). */
using viscosity_dim = dim_mult_t<pressure_dim, T>;

} // namespace dimensions

} // namespace squint

#endif // SQUINT_QUANTITY_DIMENSION_TYPES_HPP