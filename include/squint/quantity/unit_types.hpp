#ifndef SQUINT_QUANTITY_UNIT_TYPES_HPP
#define SQUINT_QUANTITY_UNIT_TYPES_HPP

#include "squint/quantity/unit.hpp"

namespace squint::units {

/// @brief Type alias for dimensionless quantities
using dimensionless = dimensionless_t<float>;

/// @brief Type alias for length quantities (base unit: meters)
using meters = meters_t<float>;
/// @brief Type alias for length quantities (feet)
using feet = feet_t<float>;
/// @brief Type alias for length quantities (inches)
using inches = inches_t<float>;
/// @brief Type alias for length quantities (kilometers)
using kilometers = kilometers_t<float>;
/// @brief Type alias for length quantities (miles)
using miles = miles_t<float>;

/// @brief Type alias for area quantities (base unit: square meters)
using square_meters = square_meters_t<float>;
/// @brief Type alias for area quantities (square feet)
using square_feet = square_feet_t<float>;
/// @brief Type alias for area quantities (square inches)
using square_inches = square_inches_t<float>;
/// @brief Type alias for area quantities (square kilometers)
using square_kilometers = square_kilometers_t<float>;
/// @brief Type alias for area quantities (square miles)
using square_miles = square_miles_t<float>;

/// @brief Type alias for volume quantities (base unit: cubic meters)
using cubic_meters = cubic_meters_t<float>;
/// @brief Type alias for volume quantities (cubic feet)
using cubic_feet = cubic_feet_t<float>;
/// @brief Type alias for volume quantities (cubic inches)
using cubic_inches = cubic_inches_t<float>;
/// @brief Type alias for volume quantities (cubic kilometers)
using cubic_kilometers = cubic_kilometers_t<float>;
/// @brief Type alias for volume quantities (cubic miles)
using cubic_miles = cubic_miles_t<float>;

/// @brief Type alias for time quantities (base unit: seconds)
using seconds = seconds_t<float>;
/// @brief Type alias for time quantities (minutes)
using minutes = minutes_t<float>;
/// @brief Type alias for time quantities (hours)
using hours = hours_t<float>;
/// @brief Type alias for time quantities (days)
using days = days_t<float>;

/// @brief Type alias for mass quantities (base unit: kilograms)
using kilograms = kilograms_t<float>;
/// @brief Type alias for mass quantities (grams)
using grams = grams_t<float>;
/// @brief Type alias for mass quantities (pounds)
using pounds = pounds_t<float>;

/// @brief Type alias for temperature quantities (base unit: kelvin)
using kelvin = kelvin_t<float>;
/// @brief Type alias for temperature quantities (celsius)
using celsius = celsius_t<float>;
/// @brief Type alias for temperature quantities (fahrenheit)
using fahrenheit = fahrenheit_t<float>;

/// @brief Type alias for angle quantities (base unit: radians)
using radians = radians_t<float>;
/// @brief Type alias for angle quantities (degrees)
using degrees = degrees_t<float>;

/// @brief Type alias for velocity quantities (base unit: meters per second)
using meters_per_second = meters_per_second_t<float>;
/// @brief Type alias for velocity quantities (kilometers per hour)
using kilometers_per_hour = kilometers_per_hour_t<float>;
/// @brief Type alias for velocity quantities (miles per hour)
using miles_per_hour = miles_per_hour_t<float>;
/// @brief Type alias for velocity quantities (feet per second)
using feet_per_second = feet_per_second_t<float>;

/// @brief Type alias for acceleration quantities (base unit: meters per second squared)
using meters_per_second_squared = meters_per_second_squared_t<float>;
/// @brief Type alias for acceleration quantities (feet per second squared)
using feet_per_second_squared = feet_per_second_squared_t<float>;

/// @brief Type alias for force quantities (base unit: newtons)
using newtons = newtons_t<float>;

/// @brief Type alias for energy quantities (base unit: joules)
using joules = joules_t<float>;

/// @brief Type alias for power quantities (base unit: watts)
using watts = watts_t<float>;

/// @brief Type alias for pressure quantities (base unit: pascals)
using pascals = pascals_t<float>;

/// @brief Type alias for electric current quantities (base unit: amperes)
using amperes = amperes_t<float>;

/// @brief Type alias for electric potential quantities (base unit: volts)
using volts = volts_t<float>;

/// @brief Type alias for electric resistance quantities (base unit: ohms)
using ohms = ohms_t<float>;

/// @brief Type alias for electric capacitance quantities (base unit: farads)
using farads = farads_t<float>;

/// @brief Type alias for magnetic flux quantities (base unit: webers)
using webers = webers_t<float>;

/// @brief Type alias for magnetic flux density quantities (base unit: teslas)
using teslas = teslas_t<float>;

} // namespace squint::units

#endif // SQUINT_QUANTITY_UNIT_TYPES_HPP