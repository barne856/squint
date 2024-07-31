#ifndef SQUINT_QUANTITY_CONSTANTS_HPP
#define SQUINT_QUANTITY_CONSTANTS_HPP

#include "squint/quantity/quantity.hpp"

#include <numbers>

namespace squint {

// Template alias for constant_quantity
template <floating_point T, typename Dimension> using constant_quantity_t = constant_quantity<T, Dimension>;

// Mathematical constants
template <floating_point T> struct math_constants {
    static constexpr auto pi = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::pi_v<T>);
    static constexpr auto e = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::e_v<T>);
    static constexpr auto sqrt2 = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::sqrt2_v<T>);
    static constexpr auto ln2 = constant_quantity_t<T, dimensions::dimensionless>(std::numbers::ln2_v<T>);
    static constexpr auto phi = constant_quantity_t<T, dimensions::dimensionless>(T(1.618033988749895));
};

// Physical constants (SI)
template <floating_point T> struct si_constants {
    // Speed of light in vacuum
    static constexpr auto c = constant_quantity_t<T, dimensions::velocity>(T(299'792'458.0));

    // Planck constant
    static constexpr auto h = constant_quantity_t<T, mult_t<dimensions::energy, dimensions::time>>(T(6.62607015e-34));

    // Reduced Planck constant (h-bar)
    static constexpr auto hbar = h / (T(2) * math_constants<T>::pi);

    // Gravitational constant
    static constexpr auto G =
        constant_quantity_t<T, div_t<mult_t<dimensions::force, dimensions::area>, pow_t<dimensions::mass, 2>>>(
            T(6.67430e-11));

    // Elementary charge
    static constexpr auto e_charge = constant_quantity_t<T, dimensions::charge>(T(1.602176634e-19));

    // Electron mass
    static constexpr auto m_e = constant_quantity_t<T, dimensions::mass>(T(9.1093837015e-31));

    // Proton mass
    static constexpr auto m_p = constant_quantity_t<T, dimensions::mass>(T(1.67262192369e-27));

    // Fine-structure constant
    static constexpr auto alpha = constant_quantity_t<T, dimensions::dimensionless>(T(7.2973525693e-3));

    // Boltzmann constant
    static constexpr auto k_B =
        constant_quantity_t<T, div_t<dimensions::energy, dimensions::temperature>>(T(1.380649e-23));

    // Avogadro constant
    static constexpr auto N_A = constant_quantity_t<T, inv_t<dimensions::amount_of_substance>>(T(6.02214076e23));

    // Gas constant
    static constexpr auto R = k_B * N_A;

    // Vacuum electric permittivity
    static constexpr auto epsilon_0 =
        constant_quantity_t<T, div_t<dimensions::capacitance, dimensions::length>>(T(8.8541878128e-12));

    // Vacuum magnetic permeability
    static constexpr auto mu_0 =
        constant_quantity_t<T, div_t<dimensions::inductance, dimensions::length>>(T(1.25663706212e-6));

    // Stefan-Boltzmann constant
    static constexpr auto sigma =
        constant_quantity_t<T, div_t<dimensions::power, mult_t<dimensions::area, pow_t<dimensions::temperature, 4>>>>(
            T(5.670374419e-8));
};

// Astronomical constants
template <floating_point T> struct astro_constants {
    // Astronomical Unit
    static constexpr auto AU = constant_quantity_t<T, dimensions::length>(T(1.495978707e11));

    // Parsec
    static constexpr auto parsec = constant_quantity_t<T, dimensions::length>(T(3.0856775814913673e16));

    // Light year
    static constexpr auto light_year =
        si_constants<T>::c * constant_quantity_t<T, dimensions::time>(T(365.25) * T(24) * T(3600));

    // Solar mass
    static constexpr auto solar_mass = constant_quantity_t<T, dimensions::mass>(T(1.988847e30));

    // Earth mass
    static constexpr auto earth_mass = constant_quantity_t<T, dimensions::mass>(T(5.97217e24));

    // Earth radius (equatorial)
    static constexpr auto earth_radius = constant_quantity_t<T, dimensions::length>(T(6.3781e6));

    // Standard gravitational acceleration on Earth
    static constexpr auto g = constant_quantity_t<T, dimensions::acceleration>(T(9.80665));
};

// Atomic and nuclear constants
template <floating_point T> struct atomic_constants {
    // Rydberg constant
    static constexpr auto R_inf = constant_quantity_t<T, dimensions::wave_number>(T(10973731.568160));

    // Bohr radius
    static constexpr auto a_0 = constant_quantity_t<T, dimensions::length>(T(5.29177210903e-11));

    // Classical electron radius
    static constexpr auto r_e = constant_quantity_t<T, dimensions::length>(T(2.8179403262e-15));

    // Proton-electron mass ratio
    static constexpr auto m_p_m_e = si_constants<T>::m_p / si_constants<T>::m_e;
};

} // namespace squint

#endif // SQUINT_QUANTITY_CONSTANTS_HPP