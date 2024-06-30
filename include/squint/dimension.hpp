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
};
// implement the concept of a dimension as a struct
template <rational Length, rational Time, rational Mass, rational Temp> struct dimension {
    using L = Length;
    using T = Time;
    using M = Mass;
    using K = Temp;
};
// multiply dimensions together
template <dimensional U1, dimensional U2> struct dim_mult {
    using type =
        dimension<std::ratio_add<typename U1::L, typename U2::L>, std::ratio_add<typename U1::T, typename U2::T>,
                  std::ratio_add<typename U1::M, typename U2::M>, std::ratio_add<typename U1::K, typename U2::K>>;
};
// divide dimensions
template <dimensional U1, dimensional U2> struct dim_div {
    using type = dimension<
        std::ratio_subtract<typename U1::L, typename U2::L>, std::ratio_subtract<typename U1::T, typename U2::T>,
        std::ratio_subtract<typename U1::M, typename U2::M>, std::ratio_subtract<typename U1::K, typename U2::K>>;
};
// raise dimension to a power
template <dimensional U, std::integral auto const N> struct dim_pow {
    using type = dimension<std::ratio_multiply<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>>;
};
// take root of a dimension
template <dimensional U, std::integral auto const N> struct dim_root {
    static_assert(N > 0, "Cannot take 0th root.");
    using type = dimension<std::ratio_divide<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>>;
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
using dimensionless = dimension<std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>>;
using length = dimension<std::ratio<1, 1>, std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>>;
using time = dimension<std::ratio<0, 1>, std::ratio<1, 1>, std::ratio<0, 1>, std::ratio<0, 1>>;
using mass = dimension<std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<1, 1>, std::ratio<0, 1>>;
using temperature = dimension<std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<1, 1>>;
using velocity = div_t<length, time>;
using acceleration = div_t<velocity, time>;
using area = mult_t<length, length>;
using volume = mult_t<area, length>;
using density = div_t<mass, volume>;
using force = mult_t<mass, acceleration>;
using force_density = mult_t<force, volume>;
using pressure = div_t<force, area>;
using dynamic_viscosity = mult_t<pressure, time>;
using kinematic_viscosity = div_t<area, time>;
using flow = div_t<volume, time>;
using energy = mult_t<force, length>;
} // namespace dimensions
} // namespace squint
#endif // SQUINT_DIMENSION_HPP