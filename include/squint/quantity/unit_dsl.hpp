/**
 * @file unit_dsl.hpp
 * @brief Defines a domain-specific language for defining units.
 *
 * This file defines a domain-specific language (DSL) for defining units. The DSL
 * provides a set of constants that represent the base units of the International
 * System of Units (SI) and other common units. These constants can be used to
 * define quantities in a more readable and expressive way.
 */
#ifndef SQUINT_QUANTITY_UNIT_DSL_HPP
#define SQUINT_QUANTITY_UNIT_DSL_HPP

#include "squint/quantity/unit_types.hpp"

namespace squint::units::dsl {

// Length units
constexpr auto m = units::meters_t<long double>{1};
constexpr auto ft = units::feet_t<long double>{1};
constexpr auto in = units::inches_t<long double>{1};
constexpr auto km = units::kilometers_t<long double>{1};
constexpr auto mi = units::miles_t<long double>{1};

// Time units
constexpr auto s = units::seconds_t<long double>{1};
constexpr auto min = units::minutes_t<long double>{1};
constexpr auto h = units::hours_t<long double>{1};
constexpr auto d = units::days_t<long double>{1};

// Mass units
constexpr auto kg = units::kilograms_t<long double>{1};
constexpr auto g = units::grams_t<long double>{1};
constexpr auto lb = units::pounds_t<long double>{1};

// Temperature units
constexpr auto K = units::kelvin_t<long double>{1};
constexpr auto C = units::celsius_t<long double>{1};
constexpr auto F = units::fahrenheit_t<long double>{1};

// Electric current units
constexpr auto A = units::amperes_t<long double>{1};

// Amount of substance units
constexpr auto mol = units::mole_t<long double>{1};
constexpr auto particle = units::particles_t<long double>{1};

// Luminous intensity units
constexpr auto cd = units::candela_t<long double>{1};

// Velocity units
constexpr auto mps = units::meters_per_second_t<long double>{1};
constexpr auto fps = units::feet_per_second_t<long double>{1};
constexpr auto kph = units::kilometers_per_hour_t<long double>{1};
constexpr auto mph = units::miles_per_hour_t<long double>{1};

// Acceleration units
constexpr auto mps2 = units::meters_per_second_squared_t<long double>{1};
constexpr auto fps2 = units::feet_per_second_squared_t<long double>{1};

// Force units
constexpr auto N = units::newtons_t<long double>{1};
constexpr auto lbf = units::pounds_force_t<long double>{1};

// Energy units
constexpr auto J = units::joules_t<long double>{1};

// Power units
constexpr auto W = units::watts_t<long double>{1};
constexpr auto hp = units::horsepower_t<long double>{1};

// Pressure units
constexpr auto Pa = units::pascals_t<long double>{1};
constexpr auto psf = units::psf_t<long double>{1};
constexpr auto psi = units::psi_t<long double>{1};
constexpr auto atm = units::atmospheres_t<long double>{1};
constexpr auto bar = units::bar_t<long double>{1};

// Electric charge units
constexpr auto coulombs = units::coulombs_t<long double>{1};

// Area units
constexpr auto m2 = units::square_meters_t<long double>{1};
constexpr auto ft2 = units::square_feet_t<long double>{1};
constexpr auto in2 = units::square_inches_t<long double>{1};
constexpr auto mi2 = units::square_miles_t<long double>{1};
constexpr auto km2 = units::square_kilometers_t<long double>{1};

// Volume units
constexpr auto m3 = units::cubic_meters_t<long double>{1};
constexpr auto ft3 = units::cubic_feet_t<long double>{1};
constexpr auto in3 = units::cubic_inches_t<long double>{1};
constexpr auto mi3 = units::cubic_miles_t<long double>{1};
constexpr auto km3 = units::cubic_kilometers_t<long double>{1};
constexpr auto L = units::liters_t<long double>{1};
constexpr auto gal = units::gallons_t<long double>{1};

// Density units
constexpr auto kg_m3 = units::kilograms_per_cubic_meter_t<long double>{1};
constexpr auto lb_ft3 = units::pounds_per_cubic_foot_t<long double>{1};

// Frequency units
constexpr auto Hz = units::hertz_t<long double>{1};

// Angle units
constexpr auto rad = units::radians_t<long double>{1};
constexpr auto deg = units::degrees_t<long double>{1};

// Angular velocity units
constexpr auto rad_s = units::radians_per_second_t<long double>{1};
constexpr auto deg_s = units::degrees_per_second_t<long double>{1};

// Angular acceleration units
constexpr auto rad_s2 = units::radians_per_second_squared_t<long double>{1};
constexpr auto deg_s2 = units::degrees_per_second_squared_t<long double>{1};

// Torque units
constexpr auto Nm = units::newton_meters_t<long double>{1};
constexpr auto ft_lb = units::foot_pounds_t<long double>{1};

// Moment of inertia units
constexpr auto kg_m2 = units::kilogram_square_meters_t<long double>{1};
constexpr auto lb_ft2 = units::pound_square_feet_t<long double>{1};

// Linear momentum units
constexpr auto kg_mps = units::kilogram_meters_per_second_t<long double>{1};
constexpr auto lb_fps = units::pound_feet_per_second_t<long double>{1};

// Voltage units
constexpr auto V = units::volts_t<long double>{1};

// Inductance units
constexpr auto H = units::henrys_t<long double>{1};

// Capacitance units
constexpr auto farads = units::farads_t<long double>{1};

// Flow units
constexpr auto m3ps = units::cubic_meters_per_second_t<long double>{1};
constexpr auto lps = units::liters_per_second_t<long double>{1};
constexpr auto gps = units::gallons_per_second_t<long double>{1};

// Viscosity units
constexpr auto Pa_s = units::pascal_seconds_t<long double>{1};
constexpr auto P = units::poise_t<long double>{1};

} // namespace squint::units::dsl

#endif // SQUINT_QUANTITY_UNIT_DSL_HPP