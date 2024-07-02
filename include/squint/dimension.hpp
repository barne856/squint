/**
 * @file dimension.hpp
 * @author Brendan Barnes
 * @brief Compile-time dimensional types. Used to define quantity types
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef SQUINT_DIMENSION_HPP
#define SQUINT_DIMENSION_HPP
#include <concepts>
#include <ratio>


namespace squint {
// define a concept for the standard library ratio template type
template <class T>
concept rational = std::is_same<T, std::ratio<T::num, T::den>>::value;
// define a concept for a dimension
template <class U>
concept dimensional = requires {
    requires rational<typename U::L>; // Length
    requires rational<typename U::T>; // Time
    requires rational<typename U::M>; // Mass
    requires rational<typename U::K>; // Temperature
    requires rational<typename U::I>; // Current
    requires rational<typename U::N>; // Amount of substance
    requires rational<typename U::J>; // Luminous intensity
};
// implement the concept of a dimension as a struct
template <rational Length, rational Time, rational Mass, rational Temp, 
          rational Current, rational AmountOfSubstance, rational LuminousIntensity>
struct dimension {
    using L = Length;
    using T = Time;
    using M = Mass;
    using K = Temp;
    using I = Current;
    using N = AmountOfSubstance;
    using J = LuminousIntensity;
};
// multiply dimensions together
template <dimensional U1, dimensional U2> struct dim_mult {
    using type = dimension<
        std::ratio_add<typename U1::L, typename U2::L>,
        std::ratio_add<typename U1::T, typename U2::T>,
        std::ratio_add<typename U1::M, typename U2::M>,
        std::ratio_add<typename U1::K, typename U2::K>,
        std::ratio_add<typename U1::I, typename U2::I>,
        std::ratio_add<typename U1::N, typename U2::N>,
        std::ratio_add<typename U1::J, typename U2::J>
    >;
};
// divide dimensions
template <dimensional U1, dimensional U2> struct dim_div {
    using type = dimension<
        std::ratio_subtract<typename U1::L, typename U2::L>,
        std::ratio_subtract<typename U1::T, typename U2::T>,
        std::ratio_subtract<typename U1::M, typename U2::M>,
        std::ratio_subtract<typename U1::K, typename U2::K>,
        std::ratio_subtract<typename U1::I, typename U2::I>,
        std::ratio_subtract<typename U1::N, typename U2::N>,
        std::ratio_subtract<typename U1::J, typename U2::J>
    >;
};

// raise dimension to a power
template <dimensional U, std::integral auto const N> struct dim_pow {
    using type = dimension<
        std::ratio_multiply<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::I, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::N, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_multiply<typename U::J, std::ratio<static_cast<std::intmax_t>(N)>>
    >;
};

// take root of a dimension
template <dimensional U, std::integral auto const N> struct dim_root {
    static_assert(N > 0, "Cannot take 0th root.");
    using type = dimension<
        std::ratio_divide<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::I, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::N, std::ratio<static_cast<std::intmax_t>(N)>>,
        std::ratio_divide<typename U::J, std::ratio<static_cast<std::intmax_t>(N)>>
    >;
};
// convenience types for combining dimensions
template <dimensional U1, dimensional U2> using mult_t = typename dim_mult<U1, U2>::type; // multiply dimensions
template <dimensional U1, dimensional U2> using div_t = typename dim_div<U1, U2>::type;   // divide dimensions
template <dimensional U, std::integral auto const N>
using pow_t = typename dim_pow<U, N>::type; // exponentiate dimensions
template <dimensional U, std::integral auto const N>
using root_t = typename dim_root<U, N>::type; // Nth root of dimensions

// common dimension definitions
namespace dimensions {
// Base dimensions
using dimensionless = dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using length = dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using time = dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using mass = dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using temperature = dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using current = dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>>;
using amount_of_substance = dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>>;
using luminous_intensity = dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>>;

using angle = dimensionless;
using solid_angle = dimensionless;
using strain = dimensionless;
using refractive_index = dimensionless;

template <dimensional U>
using inv_t = div_t<dimensionless, U>;

// Derived dimensions
using velocity = div_t<length, time>;
using acceleration = div_t<velocity, time>;
using area = mult_t<length, length>;
using volume = mult_t<area, length>;
using density = div_t<mass, volume>;
using force = mult_t<mass, acceleration>;
using force_density = div_t<force, volume>;
using pressure = div_t<force, area>;
using dynamic_viscosity = mult_t<pressure, time>;
using kinematic_viscosity = div_t<area, time>;
using flow = div_t<volume, time>;
using energy = mult_t<force, length>;
using power = div_t<energy, time>;
using charge = mult_t<current, time>;
using voltage = div_t<energy, charge>;
using capacitance = div_t<charge, voltage>;
using resistance = div_t<voltage, current>;
using conductance = inv_t<resistance>;
using magnetic_flux = mult_t<voltage, time>;
using magnetic_flux_density = div_t<magnetic_flux, area>;
using inductance = div_t<magnetic_flux, current>;
using frequency = inv_t<time>;
using angular_velocity = div_t<angle, time>;
using momentum = mult_t<mass, velocity>;
using angular_momentum = mult_t<momentum, length>;
using torque = mult_t<force, length>;
using surface_tension = div_t<force, length>;
using heat_capacity = div_t<energy, temperature>;
using specific_heat_capacity = div_t<heat_capacity, mass>;
using thermal_conductivity = div_t<power, mult_t<length, temperature>>;
using electric_field_strength = div_t<force, charge>;
using electric_displacement = div_t<charge, area>;
using permittivity = div_t<capacitance, length>;
using permeability = mult_t<inductance, inv_t<length>>;
using molar_energy = div_t<energy, amount_of_substance>;
using molar_entropy = div_t<molar_energy, temperature>;
using exposure = div_t<charge, mass>;
using dose_equivalent = div_t<energy, mass>;
using catalytic_activity = div_t<amount_of_substance, time>;
using luminance = div_t<luminous_intensity, area>;
using refractive_index = dimensionless;
using strain = dimensionless;
using angle = dimensionless;
using solid_angle = dimensionless;
using magnetic_field_strength = div_t<current, length>;
using molarity = div_t<amount_of_substance, volume>;
using molar_mass = div_t<mass, amount_of_substance>;
using impulse = mult_t<force, time>;
using wave_number = inv_t<length>;
using specific_volume = div_t<volume, mass>;
using radiant_intensity = div_t<power, solid_angle>;
using radiance = div_t<radiant_intensity, area>;
using irradiance = div_t<power, area>;
using thermal_resistance = div_t<temperature, power>;
} // namespace dimensions
} // namespace squint
#endif // SQUINT_DIMENSION_HPP