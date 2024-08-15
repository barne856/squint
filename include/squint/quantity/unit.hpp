#ifndef SQUINT_QUANTITY_UNIT_HPP
#define SQUINT_QUANTITY_UNIT_HPP

#include "squint/core/error_checking.hpp"
#include "squint/quantity/quantity.hpp"

namespace squint::units {

/**
 * @brief Base unit struct for all physical quantities.
 *
 * @tparam T The underlying numeric type (e.g., float, double).
 * @tparam D The dimension of the unit.
 * @tparam Scale The scale factor for conversion.
 * @tparam Offset The offset for conversion.
 * @tparam ErrorChecking The error checking policy.
 */
template <typename T, typename D, T Scale = T(1), T Offset = T(0),
          error_checking ErrorChecking = error_checking::disabled>
struct unit : quantity<T, D, ErrorChecking> {
    using base_quantity_type = quantity<T, D, ErrorChecking>;

    static constexpr T scale = Scale;
    static constexpr T offset = Offset;

    // Implicit constructor from base unit with the same dimension
    constexpr unit(const base_quantity_type &q) : base_quantity_type(q) {}

    // Implicit constructor from another unit with the same dimension
    template <typename U, U Scale2, U Offset2, error_checking OtherErrorChecking>
    constexpr unit(const unit<U, D, Scale2, Offset2, OtherErrorChecking> &q) : base_quantity_type(q) {}

    // Deleted constructor from different base unit
    template <typename U, typename D2, error_checking OtherErrorChecking>
    unit(const quantity<U, D2, OtherErrorChecking> &q) = delete;

    // Deleted constructor from different unit
    template <typename U, typename D2, U Scale2, U Offset2, error_checking OtherErrorChecking>
    unit(const unit<U, D2, Scale2, Offset2, OtherErrorChecking> &q) = delete;

    // Constructor from value in this unit
    constexpr explicit unit(T unit_value) : base_quantity_type((unit_value + offset) * scale) {}

    // Convert to value in this unit
    [[nodiscard]] constexpr auto unit_value() const -> T { return (this->base_quantity_type::value() / scale) - offset; }

    // Convert from base unit to this unit
    static constexpr auto convert_to(const base_quantity_type &q) -> T { return q.value() / scale - offset; }

    // Convert from this unit to base unit
    static constexpr auto convert_from(T value) -> base_quantity_type { return base_quantity_type((value + offset) * scale); }
};

/**
 * @brief Generic conversion function between units.
 *
 * @tparam ToUnit The unit type to convert to (template template parameter).
 * @tparam FromUnit The concrete unit type to convert from.
 * @param q The quantity to convert.
 * @return A new unit object of the target unit type.
 */
template <template <typename> class ToUnit, typename FromUnit> constexpr auto convert_to(const FromUnit &q) {
    return ToUnit<typename FromUnit::value_type>(q);
}

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_HPP