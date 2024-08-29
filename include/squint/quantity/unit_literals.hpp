/**
 * @file unit_literals.hpp
 * @brief Defines user-defined literals for units.
 *
 * This file provides a set of user-defined literals that allow for convenient
 * creation of quantity objects with specific units. These literals can be used
 * to express quantities in a more natural and readable way in code.
 */

#ifndef SQUINT_QUANTITY_UNIT_LITERALS_HPP
#define SQUINT_QUANTITY_UNIT_LITERALS_HPP

#include "squint/quantity/unit_types.hpp"

namespace squint::literals {
// NOLINTBEGIN
// Length literals
constexpr auto operator"" _m(long double x) { return units::meters_t<long double>(x); }
constexpr auto operator"" _km(long double x) { return units::kilometers_t<long double>(x); }
constexpr auto operator"" _cm(long double x) { return units::meters_t<long double>(x * 0.01); }
constexpr auto operator"" _mm(long double x) { return units::meters_t<long double>(x * 0.001); }
constexpr auto operator"" _um(long double x) { return units::meters_t<long double>(x * 1e-6); }
constexpr auto operator"" _nm(long double x) { return units::meters_t<long double>(x * 1e-9); }
constexpr auto operator"" _ft(long double x) { return units::feet_t<long double>(x); }
constexpr auto operator"" _in(long double x) { return units::inches_t<long double>(x); }
constexpr auto operator"" _yd(long double x) { return units::feet_t<long double>(x * 3); }
constexpr auto operator"" _mi(long double x) { return units::miles_t<long double>(x); }
constexpr auto operator"" _nmi(long double x) { return units::nautical_miles_t<long double>(x); }
constexpr auto operator"" _ly(long double x) { return units::light_years_t<long double>(x); }

// Time literals
constexpr auto operator"" _s(long double x) { return units::seconds_t<long double>(x); }
constexpr auto operator"" _ms(long double x) { return units::seconds_t<long double>(x * 0.001); }
constexpr auto operator"" _us(long double x) { return units::seconds_t<long double>(x * 1e-6); }
constexpr auto operator"" _ns(long double x) { return units::seconds_t<long double>(x * 1e-9); }
constexpr auto operator"" _min(long double x) { return units::minutes_t<long double>(x); }
constexpr auto operator"" _h(long double x) { return units::hours_t<long double>(x); }
constexpr auto operator"" _d(long double x) { return units::days_t<long double>(x); }
constexpr auto operator"" _y(long double x) { return units::years_t<long double>(x); }

// Mass literals
constexpr auto operator"" _kg(long double x) { return units::kilograms_t<long double>(x); }
constexpr auto operator"" _g(long double x) { return units::grams_t<long double>(x); }
constexpr auto operator"" _mg(long double x) { return units::kilograms_t<long double>(x * 1e-6); }
constexpr auto operator"" _ug(long double x) { return units::kilograms_t<long double>(x * 1e-9); }
constexpr auto operator"" _lb(long double x) { return units::pounds_t<long double>(x); }
constexpr auto operator"" _oz(long double x) { return units::ounces_t<long double>(x); }
constexpr auto operator"" _t(long double x) { return units::tonnes_t<long double>(x); }

// Temperature literals
constexpr auto operator"" _K(long double x) { return units::kelvin_t<long double>(x); }
constexpr auto operator"" _degC(long double x) { return units::celsius_t<long double>(x); }
constexpr auto operator"" _degF(long double x) { return units::fahrenheit_t<long double>(x); }

// Electric current literals
constexpr auto operator"" _A(long double x) { return units::amperes_t<long double>(x); }
constexpr auto operator"" _mA(long double x) { return units::amperes_t<long double>(x * 0.001); }
constexpr auto operator"" _uA(long double x) { return units::amperes_t<long double>(x * 1e-6); }

// Amount of substance literals
constexpr auto operator"" _mol(long double x) { return units::mole_t<long double>(x); }

// Luminous intensity literals
constexpr auto operator"" _cd(long double x) { return units::candela_t<long double>(x); }

// Angle literals
constexpr auto operator"" _rad(long double x) { return units::radians_t<long double>(x); }
constexpr auto operator"" _deg(long double x) { return units::degrees_t<long double>(x); }
constexpr auto operator"" _arcmin(long double x) { return units::arcminutes_t<long double>(x); }
constexpr auto operator"" _arcsec(long double x) { return units::arcseconds_t<long double>(x); }

// Velocity literals
constexpr auto operator"" _mps(long double x) { return units::meters_per_second_t<long double>(x); }
constexpr auto operator"" _kmph(long double x) { return units::kilometers_per_hour_t<long double>(x); }
constexpr auto operator"" _mph(long double x) { return units::miles_per_hour_t<long double>(x); }
constexpr auto operator"" _fps(long double x) { return units::feet_per_second_t<long double>(x); }
constexpr auto operator"" _kn(long double x) { return units::knots_t<long double>(x); }

// Acceleration literals
constexpr auto operator"" _mps2(long double x) { return units::meters_per_second_squared_t<long double>(x); }
constexpr auto operator"" _fps2(long double x) { return units::feet_per_second_squared_t<long double>(x); }
constexpr auto operator"" _g0(long double x) { return units::standard_gravity_t<long double>(x); }

// Force literals
constexpr auto operator"" _N(long double x) { return units::newtons_t<long double>(x); }
constexpr auto operator"" _lbf(long double x) { return units::pounds_force_t<long double>(x); }
constexpr auto operator"" _dyn(long double x) { return units::dynes_t<long double>(x); }

// Energy literals
constexpr auto operator"" _J(long double x) { return units::joules_t<long double>(x); }
constexpr auto operator"" _kJ(long double x) { return units::joules_t<long double>(x * 1000); }
constexpr auto operator"" _cal(long double x) { return units::calories_t<long double>(x); }
constexpr auto operator"" _kcal(long double x) { return units::kilocalories_t<long double>(x); }
constexpr auto operator"" _eV(long double x) { return units::electron_volts_t<long double>(x); }
constexpr auto operator"" _Btu(long double x) { return units::btu_t<long double>(x); }
constexpr auto operator"" _kWh(long double x) { return units::kilowatt_hours_t<long double>(x); }
constexpr auto operator"" _erg(long double x) { return units::ergs_t<long double>(x); }

// Power literals
constexpr auto operator"" _W(long double x) { return units::watts_t<long double>(x); }
constexpr auto operator"" _kW(long double x) { return units::watts_t<long double>(x * 1000); }
constexpr auto operator"" _hp(long double x) { return units::horsepower_t<long double>(x); }

// Pressure literals
constexpr auto operator"" _Pa(long double x) { return units::pascals_t<long double>(x); }
constexpr auto operator"" _kPa(long double x) { return units::pascals_t<long double>(x * 1000); }
constexpr auto operator"" _MPa(long double x) { return units::pascals_t<long double>(x * 1e6); }
constexpr auto operator"" _bar(long double x) { return units::bars_t<long double>(x); }
constexpr auto operator"" _atm(long double x) { return units::atmospheres_t<long double>(x); }
constexpr auto operator"" _mmHg(long double x) { return units::millimeters_of_mercury_t<long double>(x); }
constexpr auto operator"" _psi(long double x) { return units::pounds_per_square_inch_t<long double>(x); }

// Electric charge literals
constexpr auto operator"" _C(long double x) { return units::coulombs_t<long double>(x); }
constexpr auto operator"" _Ah(long double x) { return units::ampere_hours_t<long double>(x); }

// Electric potential literals
constexpr auto operator"" _V(long double x) { return units::volts_t<long double>(x); }
constexpr auto operator"" _mV(long double x) { return units::volts_t<long double>(x * 0.001); }
constexpr auto operator"" _kV(long double x) { return units::volts_t<long double>(x * 1000); }

// Capacitance literals
constexpr auto operator"" _F(long double x) { return units::farads_t<long double>(x); }
constexpr auto operator"" _uF(long double x) { return units::farads_t<long double>(x * 1e-6); }
constexpr auto operator"" _nF(long double x) { return units::farads_t<long double>(x * 1e-9); }
constexpr auto operator"" _pF(long double x) { return units::farads_t<long double>(x * 1e-12); }

// Resistance literals
constexpr auto operator"" _ohm(long double x) { return units::ohms_t<long double>(x); }
constexpr auto operator"" _kohm(long double x) { return units::ohms_t<long double>(x * 1000); }
constexpr auto operator"" _Mohm(long double x) { return units::ohms_t<long double>(x * 1e6); }

// Conductance literals
constexpr auto operator"" _S(long double x) { return units::siemens_t<long double>(x); }
constexpr auto operator"" _mS(long double x) { return units::siemens_t<long double>(x * 0.001); }
constexpr auto operator"" _uS(long double x) { return units::siemens_t<long double>(x * 1e-6); }

// Magnetic flux literals
constexpr auto operator"" _Wb(long double x) { return units::webers_t<long double>(x); }
constexpr auto operator"" _Mx(long double x) { return units::maxwells_t<long double>(x); }

// Magnetic flux density literals
constexpr auto operator"" _T(long double x) { return units::teslas_t<long double>(x); }
constexpr auto operator"" _G(long double x) { return units::gauss_t<long double>(x); }

// Inductance literals
constexpr auto operator"" _H(long double x) { return units::henries_t<long double>(x); }
constexpr auto operator"" _mH(long double x) { return units::henries_t<long double>(x * 0.001); }
constexpr auto operator"" _uH(long double x) { return units::henries_t<long double>(x * 1e-6); }

// Area literals
constexpr auto operator"" _m2(long double x) { return units::square_meters_t<long double>(x); }
constexpr auto operator"" _cm2(long double x) { return units::square_meters_t<long double>(x * 1e-4); }
constexpr auto operator"" _mm2(long double x) { return units::square_meters_t<long double>(x * 1e-6); }
constexpr auto operator"" _ft2(long double x) { return units::square_feet_t<long double>(x); }
constexpr auto operator"" _in2(long double x) { return units::square_inches_t<long double>(x); }
constexpr auto operator"" _km2(long double x) { return units::square_kilometers_t<long double>(x); }
constexpr auto operator"" _mi2(long double x) { return units::square_miles_t<long double>(x); }
constexpr auto operator"" _ha(long double x) { return units::hectares_t<long double>(x); }
constexpr auto operator"" _acre(long double x) { return units::acres_t<long double>(x); }

// Volume literals
constexpr auto operator"" _m3(long double x) { return units::cubic_meters_t<long double>(x); }
constexpr auto operator"" _cm3(long double x) { return units::cubic_meters_t<long double>(x * 1e-6); }
constexpr auto operator"" _mm3(long double x) { return units::cubic_meters_t<long double>(x * 1e-9); }
constexpr auto operator"" _L(long double x) { return units::liters_t<long double>(x); }
constexpr auto operator"" _mL(long double x) { return units::liters_t<long double>(x * 0.001); }
constexpr auto operator"" _ft3(long double x) { return units::cubic_feet_t<long double>(x); }
constexpr auto operator"" _in3(long double x) { return units::cubic_inches_t<long double>(x); }
constexpr auto operator"" _gal(long double x) { return units::gallons_t<long double>(x); }

// Frequency literals
constexpr auto operator"" _Hz(long double x) { return units::hertz_t<long double>(x); }
constexpr auto operator"" _kHz(long double x) { return units::kilohertz_t<long double>(x); }
constexpr auto operator"" _MHz(long double x) { return units::megahertz_t<long double>(x); }
constexpr auto operator"" _GHz(long double x) { return units::gigahertz_t<long double>(x); }

// Angular velocity literals
constexpr auto operator"" _radps(long double x) { return units::radians_per_second_t<long double>(x); }
constexpr auto operator"" _degps(long double x) { return units::degrees_per_second_t<long double>(x); }
constexpr auto operator"" _rpm(long double x) { return units::revolutions_per_minute_t<long double>(x); }

// Angular acceleration literals
constexpr auto operator"" _radps2(long double x) { return units::radians_per_second_squared_t<long double>(x); }
constexpr auto operator"" _degps2(long double x) { return units::degrees_per_second_squared_t<long double>(x); }

// Torque literals
constexpr auto operator"" _Nm(long double x) { return units::newton_meters_t<long double>(x); }
constexpr auto operator"" _ftlb(long double x) { return units::foot_pounds_t<long double>(x); }

// Moment of inertia literals
constexpr auto operator"" _kgm2(long double x) { return units::kilogram_square_meters_t<long double>(x); }

// Dynamic viscosity literals
constexpr auto operator"" _Pas(long double x) { return units::pascal_seconds_t<long double>(x); }
constexpr auto operator"" _P(long double x) { return units::poise_t<long double>(x); }

// Kinematic viscosity literals
constexpr auto operator"" _m2ps(long double x) { return units::square_meters_per_second_t<long double>(x); }
constexpr auto operator"" _St(long double x) { return units::stokes_t<long double>(x); }

// Surface tension literals
constexpr auto operator"" _Npm(long double x) { return units::newtons_per_meter_t<long double>(x); }
constexpr auto operator"" _dyncm(long double x) { return units::dynes_per_centimeter_t<long double>(x); }

// Specific heat capacity literals
constexpr auto operator"" _JpkgK(long double x) { return units::joules_per_kilogram_kelvin_t<long double>(x); }
constexpr auto operator"" _calpgC(long double x) { return units::calories_per_gram_celsius_t<long double>(x); }

// Thermal conductivity literals
constexpr auto operator"" _WpmK(long double x) { return units::watts_per_meter_kelvin_t<long double>(x); }

// Electric field strength literals
constexpr auto operator"" _Vpm(long double x) { return units::volts_per_meter_t<long double>(x); }

// Permittivity literals
constexpr auto operator"" _Fpm(long double x) { return units::farads_per_meter_t<long double>(x); }

// Permeability literals
constexpr auto operator"" _Hpm(long double x) { return units::henries_per_meter_t<long double>(x); }

// Absorbed dose literals
constexpr auto operator"" _Gy(long double x) { return units::grays_t<long double>(x); }

// Equivalent dose literals
constexpr auto operator"" _Sv(long double x) { return units::sieverts_t<long double>(x); }
constexpr auto operator"" _rem(long double x) { return units::rems_t<long double>(x); }

// Catalytic activity literals
constexpr auto operator"" _kat(long double x) { return units::katal_t<long double>(x); }
constexpr auto operator"" _U(long double x) { return units::enzyme_unit_t<long double>(x); }

// Concentration literals
constexpr auto operator"" _molpm3(long double x) { return units::moles_per_cubic_meter_t<long double>(x); }
constexpr auto operator"" _molpL(long double x) { return units::moles_per_liter_t<long double>(x); }

// Molality literals
constexpr auto operator"" _molpkg(long double x) { return units::moles_per_kilogram_t<long double>(x); }

// Molar mass literals
constexpr auto operator"" _kgpmol(long double x) { return units::kilograms_per_mole_t<long double>(x); }
constexpr auto operator"" _gpmol(long double x) { return units::grams_per_mole_t<long double>(x); }

// Luminous flux literals
constexpr auto operator"" _lm(long double x) { return units::lumens_t<long double>(x); }

// Illuminance literals
constexpr auto operator"" _lx(long double x) { return units::lux_t<long double>(x); }
constexpr auto operator"" _fc(long double x) { return units::foot_candles_t<long double>(x); }

// Luminous energy literals
constexpr auto operator"" _lms(long double x) { return units::lumen_seconds_t<long double>(x); }

// Luminous exposure literals
constexpr auto operator"" _lxs(long double x) { return units::lux_seconds_t<long double>(x); }

// Radioactivity literals
constexpr auto operator"" _Bq(long double x) { return units::becquerels_t<long double>(x); }
constexpr auto operator"" _Ci(long double x) { return units::curies_t<long double>(x); }

// Flow rate literals
constexpr auto operator"" _m3ps(long double x) { return units::cubic_meters_per_second_t<long double>(x); }
constexpr auto operator"" _Lps(long double x) { return units::liters_per_second_t<long double>(x); }
constexpr auto operator"" _galpm(long double x) { return units::gallons_per_minute_t<long double>(x); }

// Integer literal operators
// Note: These are provided for convenience, but may lose precision for large values

#define SQUINT_DEFINE_INTEGER_LITERAL(unit, type)                                                                      \
    constexpr auto operator"" _##unit(unsigned long long int x) {                                                      \
        return units::type##_t<long double>(static_cast<long double>(x));                                              \
    }

SQUINT_DEFINE_INTEGER_LITERAL(m, meters)
SQUINT_DEFINE_INTEGER_LITERAL(km, kilometers)
SQUINT_DEFINE_INTEGER_LITERAL(cm, meters) // Note: uses meters_t with a factor
SQUINT_DEFINE_INTEGER_LITERAL(mm, meters) // Note: uses meters_t with a factor
SQUINT_DEFINE_INTEGER_LITERAL(ft, feet)
SQUINT_DEFINE_INTEGER_LITERAL(in, inches)
SQUINT_DEFINE_INTEGER_LITERAL(mi, miles)
SQUINT_DEFINE_INTEGER_LITERAL(nmi, nautical_miles)
SQUINT_DEFINE_INTEGER_LITERAL(s, seconds)
SQUINT_DEFINE_INTEGER_LITERAL(min, minutes)
SQUINT_DEFINE_INTEGER_LITERAL(h, hours)
SQUINT_DEFINE_INTEGER_LITERAL(d, days)
SQUINT_DEFINE_INTEGER_LITERAL(y, years)
SQUINT_DEFINE_INTEGER_LITERAL(kg, kilograms)
SQUINT_DEFINE_INTEGER_LITERAL(g, grams)
SQUINT_DEFINE_INTEGER_LITERAL(lb, pounds)
SQUINT_DEFINE_INTEGER_LITERAL(oz, ounces)
SQUINT_DEFINE_INTEGER_LITERAL(t, tonnes)
SQUINT_DEFINE_INTEGER_LITERAL(K, kelvin)
SQUINT_DEFINE_INTEGER_LITERAL(A, amperes)
SQUINT_DEFINE_INTEGER_LITERAL(mol, mole)
SQUINT_DEFINE_INTEGER_LITERAL(cd, candela)
SQUINT_DEFINE_INTEGER_LITERAL(rad, radians)
SQUINT_DEFINE_INTEGER_LITERAL(deg, degrees)
SQUINT_DEFINE_INTEGER_LITERAL(Hz, hertz)
SQUINT_DEFINE_INTEGER_LITERAL(N, newtons)
SQUINT_DEFINE_INTEGER_LITERAL(Pa, pascals)
SQUINT_DEFINE_INTEGER_LITERAL(J, joules)
SQUINT_DEFINE_INTEGER_LITERAL(W, watts)
SQUINT_DEFINE_INTEGER_LITERAL(C, coulombs)
SQUINT_DEFINE_INTEGER_LITERAL(V, volts)
SQUINT_DEFINE_INTEGER_LITERAL(F, farads)
SQUINT_DEFINE_INTEGER_LITERAL(ohm, ohms)
SQUINT_DEFINE_INTEGER_LITERAL(S, siemens)
SQUINT_DEFINE_INTEGER_LITERAL(Wb, webers)
SQUINT_DEFINE_INTEGER_LITERAL(T, teslas)
SQUINT_DEFINE_INTEGER_LITERAL(H, henries)
SQUINT_DEFINE_INTEGER_LITERAL(lm, lumens)
SQUINT_DEFINE_INTEGER_LITERAL(lx, lux)
SQUINT_DEFINE_INTEGER_LITERAL(Bq, becquerels)
SQUINT_DEFINE_INTEGER_LITERAL(Gy, grays)
SQUINT_DEFINE_INTEGER_LITERAL(Sv, sieverts)
SQUINT_DEFINE_INTEGER_LITERAL(kat, katal)

// For units that don't fit the pattern, we define them individually
constexpr auto operator"" _degC(unsigned long long int x) {
    return units::celsius_t<long double>(static_cast<long double>(x));
}
constexpr auto operator"" _degF(unsigned long long int x) {
    return units::fahrenheit_t<long double>(static_cast<long double>(x));
}
constexpr auto operator"" _eV(unsigned long long int x) {
    return units::electron_volts_t<long double>(static_cast<long double>(x));
}
constexpr auto operator"" _L(unsigned long long int x) {
    return units::liters_t<long double>(static_cast<long double>(x));
}

#undef SQUINT_DEFINE_INTEGER_LITERAL
// NOLINTEND
} // namespace squint::literals

#endif // SQUINT_QUANTITY_UNIT_LITERALS_HPP