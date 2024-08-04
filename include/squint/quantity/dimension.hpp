/**
 * @file dimension.hpp
 * @brief Defines operations for handling physical dimensions.
 *
 * This file provides the core functionality for representing and manipulating
 * physical dimensions in a type-safe manner. It uses the C++ standard library's
 * ratio type to represent dimensional exponents and defines operations for
 * combining dimensions through multiplication, division, powers, and roots.
 *
 * The dimensions are based on the seven SI base units: length, time, mass,
 * temperature, electric current, amount of substance, and luminous intensity.
 */

#ifndef SQUINT_QUANTITY_DIMENSION_HPP
#define SQUINT_QUANTITY_DIMENSION_HPP

#include "squint/core/concepts.hpp"


namespace squint {

/**
 * @brief Represents a physical dimension.
 *
 * This struct template represents a physical dimension as a combination of
 * rational exponents for each of the seven SI base dimensions.
 *
 * @tparam Length Exponent for length dimension.
 * @tparam Time Exponent for time dimension.
 * @tparam Mass Exponent for mass dimension.
 * @tparam Temp Exponent for temperature dimension.
 * @tparam Current Exponent for electric current dimension.
 * @tparam AmountOfSubstance Exponent for amount of substance dimension.
 * @tparam LuminousIntensity Exponent for luminous intensity dimension.
 */
template <rational Length, rational Time, rational Mass, rational Temp, rational Current, rational AmountOfSubstance,
          rational LuminousIntensity>
struct dimension {
    using L = Length;
    using T = Time;
    using M = Mass;
    using K = Temp;
    using I = Current;
    using N = AmountOfSubstance;
    using J = LuminousIntensity;
};

/**
 * @brief Multiplies two dimensions.
 *
 * This struct template provides a type alias for the result of multiplying
 * two dimensions. The exponents of each base dimension are added.
 *
 * @tparam U1 The first dimension.
 * @tparam U2 The second dimension.
 */
template <dimensional U1, dimensional U2> struct dim_mult {
    using type =
        dimension<std::ratio_add<typename U1::L, typename U2::L>, std::ratio_add<typename U1::T, typename U2::T>,
                  std::ratio_add<typename U1::M, typename U2::M>, std::ratio_add<typename U1::K, typename U2::K>,
                  std::ratio_add<typename U1::I, typename U2::I>, std::ratio_add<typename U1::N, typename U2::N>,
                  std::ratio_add<typename U1::J, typename U2::J>>;
};

/**
 * @brief Divides two dimensions.
 *
 * This struct template provides a type alias for the result of dividing
 * two dimensions. The exponents of each base dimension of the second dimension
 * are subtracted from those of the first dimension.
 *
 * @tparam U1 The dividend dimension.
 * @tparam U2 The divisor dimension.
 */
template <dimensional U1, dimensional U2> struct dim_div {
    using type = dimension<
        std::ratio_subtract<typename U1::L, typename U2::L>, std::ratio_subtract<typename U1::T, typename U2::T>,
        std::ratio_subtract<typename U1::M, typename U2::M>, std::ratio_subtract<typename U1::K, typename U2::K>,
        std::ratio_subtract<typename U1::I, typename U2::I>, std::ratio_subtract<typename U1::N, typename U2::N>,
        std::ratio_subtract<typename U1::J, typename U2::J>>;
};

/**
 * @brief Raises a dimension to an integral power.
 *
 * This struct template provides a type alias for the result of raising
 * a dimension to an integral power. Each base dimension's exponent is
 * multiplied by the power.
 *
 * @tparam U The dimension to be raised to a power.
 * @tparam N The integral power to raise the dimension to.
 */
template <dimensional U, std::integral auto const N> struct dim_pow {
    using type = dimension<std::ratio_multiply<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::I, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::N, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_multiply<typename U::J, std::ratio<static_cast<std::intmax_t>(N)>>>;
};

/**
 * @brief Takes the root of a dimension.
 *
 * This struct template provides a type alias for the result of taking
 * the Nth root of a dimension. Each base dimension's exponent is
 * divided by N.
 *
 * @tparam U The dimension to take the root of.
 * @tparam N The root to take (must be positive).
 */
template <dimensional U, std::integral auto const N> struct dim_root {
    static_assert(N > 0, "Cannot take 0th root.");
    using type = dimension<std::ratio_divide<typename U::L, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::T, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::M, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::K, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::I, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::N, std::ratio<static_cast<std::intmax_t>(N)>>,
                           std::ratio_divide<typename U::J, std::ratio<static_cast<std::intmax_t>(N)>>>;
};

} // namespace squint

#endif // SQUINT_QUANTITY_DIMENSION_HPP