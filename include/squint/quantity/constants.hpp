/**
 * @file constants.hpp
 * @brief Defines physical and mathematical constants as dimensioned quantities.
 *
 * This file provides a set of commonly used constants in physics and mathematics,
 * represented as dimensioned quantities. The constants are organized into different
 * categories: mathematical, SI (Système International), astronomical, and atomic/nuclear.
 *
 * All constants are defined as static constexpr members of their respective structs,
 * parameterized by a floating-point type T for flexibility in precision.
 */

#ifndef SQUINT_QUANTITY_CONSTANTS_HPP
#define SQUINT_QUANTITY_CONSTANTS_HPP

#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/quantity_types.hpp"

#include <numbers>

namespace squint {

/**
 * @brief Mathematical constants.
 *
 * Provides fundamental mathematical constants as dimensionless quantities.
 *
 * @tparam T The underlying floating-point type for the constants.
 */
template <floating_point T> struct math_constants {
    /// @brief The ratio of a circle's circumference to its diameter.
    static constexpr auto pi = constant_quantity_t<T, dimensions::unity>(std::numbers::pi_v<T>);

    /// @brief The base of natural logarithms.
    static constexpr auto e = constant_quantity_t<T, dimensions::unity>(std::numbers::e_v<T>);

    /// @brief The square root of 2.
    static constexpr auto sqrt2 = constant_quantity_t<T, dimensions::unity>(std::numbers::sqrt2_v<T>);

    /// @brief The natural logarithm of 2.
    static constexpr auto ln2 = constant_quantity_t<T, dimensions::unity>(std::numbers::ln2_v<T>);

    /// @brief The golden ratio.
    static constexpr auto phi = constant_quantity_t<T, dimensions::unity>(T(1.618033988749895));
};

/**
 * @brief Physical constants in the SI system.
 *
 * Provides fundamental physical constants as dimensioned quantities in the SI system.
 *
 * @tparam T The underlying floating-point type for the constants.
 */
template <floating_point T> struct si_constants {
    /// @brief Speed of light in vacuum (m/s).
    static constexpr auto c = constant_quantity_t<T, dimensions::velocity_dim>(T(299'792'458.0));

    /// @brief Planck constant (J⋅s).
    static constexpr auto h =
        constant_quantity_t<T, dim_mult_t<dimensions::energy_dim, dimensions::T>>(T(6.62607015e-34));

    /// @brief Reduced Planck constant (J⋅s).
    static constexpr auto hbar = h / (T(2) * math_constants<T>::pi);

    /// @brief Gravitational constant (m³/(kg⋅s²)).
    static constexpr auto G = constant_quantity_t<
        T, dim_div_t<dim_mult_t<dimensions::force_dim, dimensions::area_dim>, dim_pow_t<dimensions::M, 2>>>(
        T(6.67430e-11));

    /// @brief Elementary charge (C).
    static constexpr auto e_charge = constant_quantity_t<T, dimensions::charge_dim>(T(1.602176634e-19));

    /// @brief Electron mass (kg).
    static constexpr auto m_e = constant_quantity_t<T, dimensions::M>(T(9.1093837015e-31));

    /// @brief Proton mass (kg).
    static constexpr auto m_p = constant_quantity_t<T, dimensions::M>(T(1.67262192369e-27));

    /// @brief Fine-structure constant (dimensionless).
    static constexpr auto alpha = constant_quantity_t<T, dimensions::unity>(T(7.2973525693e-3));

    /// @brief Boltzmann constant (J/K).
    static constexpr auto k_B =
        constant_quantity_t<T, dim_div_t<dimensions::energy_dim, dimensions::K>>(T(1.380649e-23));

    /// @brief Avogadro constant (mol^-1).
    static constexpr auto N_A = constant_quantity_t<T, dim_inv_t<dimensions::N>>(T(6.02214076e23));

    /// @brief Gas constant (J/(mol⋅K)).
    static constexpr auto R = k_B * N_A;

    /// @brief Vacuum electric permittivity (F/m).
    static constexpr auto epsilon_0 =
        constant_quantity_t<T, dim_div_t<dimensions::capacitance_dim, dimensions::L>>(T(8.8541878128e-12));

    /// @brief Vacuum magnetic permeability (H/m).
    static constexpr auto mu_0 =
        constant_quantity_t<T, dim_div_t<dimensions::inductance_dim, dimensions::L>>(T(1.25663706212e-6));

    /// @brief Stefan-Boltzmann constant (W/(m²⋅K⁴)).
    static constexpr auto sigma = constant_quantity_t<
        T, dim_div_t<dimensions::power_dim, dim_mult_t<dimensions::area_dim, dim_pow_t<dimensions::K, 4>>>>(
        T(5.670374419e-8));
};

/**
 * @brief Astronomical constants.
 *
 * Provides commonly used astronomical constants as dimensioned quantities.
 *
 * @tparam T The underlying floating-point type for the constants.
 */
template <floating_point T> struct astro_constants {
    /// @brief Astronomical Unit (m).
    static constexpr auto AU = constant_quantity_t<T, dimensions::L>(T(1.495978707e11));

    /// @brief Parsec (m).
    static constexpr auto parsec = constant_quantity_t<T, dimensions::L>(T(3.0856775814913673e16));

    /// @brief Light year (m).
    static constexpr auto light_year =
        si_constants<T>::c * constant_quantity_t<T, dimensions::T>(T(365.25) * T(24) * T(3600));

    /// @brief Solar mass (kg).
    static constexpr auto solar_mass = constant_quantity_t<T, dimensions::M>(T(1.988847e30));

    /// @brief Earth mass (kg).
    static constexpr auto earth_mass = constant_quantity_t<T, dimensions::M>(T(5.97217e24));

    /// @brief Earth radius (equatorial) (m).
    static constexpr auto earth_radius = constant_quantity_t<T, dimensions::L>(T(6.3781e6));

    /// @brief Standard gravitational acceleration on Earth (m/s²).
    static constexpr auto g = constant_quantity_t<T, dimensions::acceleration_dim>(T(9.80665));
};

/**
 * @brief Atomic and nuclear constants.
 *
 * Provides constants related to atomic and nuclear physics as dimensioned quantities.
 *
 * @tparam T The underlying floating-point type for the constants.
 */
template <floating_point T> struct atomic_constants {
    /// @brief Rydberg constant (m^-1).
    static constexpr auto R_inf = constant_quantity_t<T, dimensions::frequency_dim>(T(10973731.568160));

    /// @brief Bohr radius (m).
    static constexpr auto a_0 = constant_quantity_t<T, dimensions::L>(T(5.29177210903e-11));

    /// @brief Classical electron radius (m).
    static constexpr auto r_e = constant_quantity_t<T, dimensions::L>(T(2.8179403262e-15));

    /// @brief Proton-electron mass ratio (dimensionless).
    static constexpr auto m_p_m_e = si_constants<T>::m_p / si_constants<T>::m_e;
};

} // namespace squint

#endif // SQUINT_QUANTITY_CONSTANTS_HPP