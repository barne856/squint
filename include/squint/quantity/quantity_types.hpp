/**
 * @file quantity_types.hpp
 * @brief Defines type aliases for various quantity specializations.
 *
 * This file provides convenient type aliases for quantities with different
 * error checking policies and for constant quantities.
 */

#ifndef SQUINT_QUANTITY_QUANTITY_TYPES_HPP
#define SQUINT_QUANTITY_QUANTITY_TYPES_HPP

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

} // namespace squint

#endif // SQUINT_QUANTITY_QUANTITY_TYPES_HPP