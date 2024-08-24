#ifndef SQUINT_QUANTITY_UNIT_LITERALS_HPP
#define SQUINT_QUANTITY_UNIT_LITERALS_HPP

#include "squint/quantity/unit_types.hpp"

namespace squint::literals {

// Length literals
constexpr auto operator"" _m(long double x) { return units::meters_t<long double>(x); }
constexpr auto operator"" _km(long double x) { return units::kilometers_t<long double>(x); }
constexpr auto operator"" _in(long double x) { return units::inches_t<long double>(x); }
constexpr auto operator"" _ft(long double x) { return units::feet_t<long double>(x); }
constexpr auto operator"" _mi(long double x) { return units::miles_t<long double>(x); }

// Time literals
constexpr auto operator"" _s(long double x) { return units::seconds_t<long double>(x); }
constexpr auto operator"" _min(long double x) { return units::minutes_t<long double>(x); }
constexpr auto operator"" _h(long double x) { return units::hours_t<long double>(x); }
constexpr auto operator"" _d(long double x) { return units::days_t<long double>(x); }

// Mass literals
constexpr auto operator"" _kg(long double x) { return units::kilograms_t<long double>(x); }
constexpr auto operator"" _g(long double x) { return units::grams_t<long double>(x); }
constexpr auto operator"" _lb(long double x) { return units::pounds_t<long double>(x); }

// Temperature literals
constexpr auto operator"" _K(long double x) { return units::kelvin_t<long double>(x); }
constexpr auto operator"" _C(long double x) { return units::celsius_t<long double>(x); }
constexpr auto operator"" _F(long double x) { return units::fahrenheit_t<long double>(x); }

// Angle literals
constexpr auto operator"" _rad(long double x) { return units::radians_t<long double>(x); }
constexpr auto operator"" _deg(long double x) { return units::degrees_t<long double>(x); }

// Force literals
constexpr auto operator"" _N(long double x) { return units::newtons_t<long double>(x); }
constexpr auto operator"" _lbf(long double x) { return units::pounds_force_t<long double>(x); }

// Energy literals
constexpr auto operator"" _J(long double x) { return units::joules_t<long double>(x); }

// Power literals
constexpr auto operator"" _W(long double x) { return units::watts_t<long double>(x); }
constexpr auto operator"" _hp(long double x) { return units::horsepower_t<long double>(x); }

// Pressure literals
constexpr auto operator"" _Pa(long double x) { return units::pascals_t<long double>(x); }
constexpr auto operator"" _bar(long double x) { return units::bar_t<long double>(x); }
constexpr auto operator"" _atm(long double x) { return units::atmospheres_t<long double>(x); }
constexpr auto operator"" _psi(long double x) { return units::psi_t<long double>(x); }

// Volume literals
constexpr auto operator"" _L(long double x) { return units::liters_t<long double>(x); }
constexpr auto operator"" _gal(long double x) { return units::gallons_t<long double>(x); }

// Frequency literals
constexpr auto operator"" _Hz(long double x) { return units::hertz_t<long double>(x); }

// Velocity literals
constexpr auto operator"" _mps(long double x) { return units::meters_per_second_t<long double>(x); }
constexpr auto operator"" _kph(long double x) { return units::kilometers_per_hour_t<long double>(x); }
constexpr auto operator"" _mph(long double x) { return units::miles_per_hour_t<long double>(x); }

// Electric current literals
constexpr auto operator"" _A(long double x) { return units::amperes_t<long double>(x); }

// Electric potential literals
constexpr auto operator"" _V(long double x) { return units::volts_t<long double>(x); }

// Inductance literals
constexpr auto operator"" _H(long double x) { return units::henrys_t<long double>(x); }

} // namespace squint::literals

#endif // SQUINT_QUANTITY_UNIT_LITERALS_HPP