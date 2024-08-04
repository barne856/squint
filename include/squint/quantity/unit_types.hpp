#ifndef SQUINT_QUANTITY_UNITS_TYPES_HPP
#define SQUINT_QUANTITY_UNITS_TYPES_HPP

#include "squint/quantity/unit.hpp"

namespace squint::units {

/// @brief Type alias for length quantities (base unit: meters)
using length = decltype(meters(float{}));

/// @brief Type alias for time quantities (base unit: seconds)
using time = decltype(seconds(float{}));

/// @brief Type alias for mass quantities (base unit: kilograms)
using mass = decltype(kilograms(float{}));

/// @brief Type alias for temperature quantities (base unit: kelvin)
using temperature = decltype(kelvin(float{}));

/// @brief Type alias for angle quantities (base unit: radians)
using angle = decltype(radians(float{}));

/// @brief Type alias for velocity quantities (base unit: meters per second)
using velocity = decltype(meters_per_second(float{}));

/// @brief Type alias for acceleration quantities (base unit: meters per second squared)
using acceleration = decltype(meters_per_second_squared(float{}));

/// @brief Type alias for force quantities (base unit: newtons)
using force = decltype(newtons(float{}));

/// @brief Type alias for energy quantities (base unit: joules)
using energy = decltype(joules(float{}));

/// @brief Type alias for power quantities (base unit: watts)
using power = decltype(watts(float{}));

/// @brief Type alias for pressure quantities (base unit: pascals)
using pressure = decltype(pascals(float{}));

/// @brief Type alias for electric current quantities (base unit: amperes)
using current = decltype(amperes(float{}));

/// @brief Type alias for electric potential quantities (base unit: volts)
using voltage = decltype(volts(float{}));

/// @brief Type alias for electric resistance quantities (base unit: ohms)
using resistance = decltype(ohms(float{}));

/// @brief Type alias for electric capacitance quantities (base unit: farads)
using capacitance = decltype(farads(float{}));

/// @brief Type alias for magnetic flux quantities (base unit: webers)
using magnetic_flux = decltype(webers(float{}));

/// @brief Type alias for magnetic flux density quantities (base unit: teslas)
using magnetic_flux_density = decltype(teslas(float{}));

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNITS_TYPES_HPP