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
constexpr auto km = units::kilometers_t<long double>{1};
constexpr auto cm = units::meters_t<long double>{0.01};
constexpr auto mm = units::meters_t<long double>{0.001};
constexpr auto um = units::meters_t<long double>{1e-6};
constexpr auto nm = units::meters_t<long double>{1e-9};
constexpr auto ft = units::feet_t<long double>{1};
constexpr auto in = units::inches_t<long double>{1};
constexpr auto yd = units::feet_t<long double>{3};
constexpr auto mi = units::miles_t<long double>{1};
constexpr auto nmi = units::nautical_miles_t<long double>{1};
constexpr auto ly = units::light_years_t<long double>{1};

// Time units
constexpr auto s = units::seconds_t<long double>{1};
constexpr auto ms = units::seconds_t<long double>{0.001};
constexpr auto us = units::seconds_t<long double>{1e-6};
constexpr auto ns = units::seconds_t<long double>{1e-9};
constexpr auto min = units::minutes_t<long double>{1};
constexpr auto h = units::hours_t<long double>{1};
constexpr auto d = units::days_t<long double>{1};
constexpr auto y = units::years_t<long double>{1};

// Mass units
constexpr auto kg = units::kilograms_t<long double>{1};
constexpr auto g = units::grams_t<long double>{1};
constexpr auto mg = units::kilograms_t<long double>{1e-6};
constexpr auto ug = units::kilograms_t<long double>{1e-9};
constexpr auto lb = units::pounds_t<long double>{1};
constexpr auto oz = units::ounces_t<long double>{1};
constexpr auto t = units::tonnes_t<long double>{1};

// Temperature units
constexpr auto K = units::kelvin_t<long double>{1};
constexpr auto degC = units::celsius_t<long double>{1};
constexpr auto degF = units::fahrenheit_t<long double>{1};

// Electric current units
constexpr auto A = units::amperes_t<long double>{1};
constexpr auto mA = units::amperes_t<long double>{0.001};
constexpr auto uA = units::amperes_t<long double>{1e-6};

// Amount of substance units
constexpr auto mol = units::mole_t<long double>{1};

// Luminous intensity units
constexpr auto cd = units::candela_t<long double>{1};

// Angle units
constexpr auto rad = units::radians_t<long double>{1};
constexpr auto deg = units::degrees_t<long double>{1};
constexpr auto arcmin = units::arcminutes_t<long double>{1};
constexpr auto arcsec = units::arcseconds_t<long double>{1};

// Velocity units
constexpr auto mps = units::meters_per_second_t<long double>{1};
constexpr auto kmph = units::kilometers_per_hour_t<long double>{1};
constexpr auto mph = units::miles_per_hour_t<long double>{1};
constexpr auto fps = units::feet_per_second_t<long double>{1};
constexpr auto kn = units::knots_t<long double>{1};

// Acceleration units
constexpr auto mps2 = units::meters_per_second_squared_t<long double>{1};
constexpr auto fps2 = units::feet_per_second_squared_t<long double>{1};
constexpr auto g0 = units::standard_gravity_t<long double>{1};

// Force units
constexpr auto N = units::newtons_t<long double>{1};
constexpr auto lbf = units::pounds_force_t<long double>{1};
constexpr auto dyn = units::dynes_t<long double>{1};

// Energy units
constexpr auto J = units::joules_t<long double>{1};
constexpr auto kJ = units::joules_t<long double>{1000};
constexpr auto cal = units::calories_t<long double>{1};
constexpr auto kcal = units::kilocalories_t<long double>{1};
constexpr auto eV = units::electron_volts_t<long double>{1};
constexpr auto Btu = units::btu_t<long double>{1};
constexpr auto kWh = units::kilowatt_hours_t<long double>{1};
constexpr auto erg = units::ergs_t<long double>{1};

// Power units
constexpr auto W = units::watts_t<long double>{1};
constexpr auto kW = units::watts_t<long double>{1000};
constexpr auto hp = units::horsepower_t<long double>{1};

// Pressure units
constexpr auto Pa = units::pascals_t<long double>{1};
constexpr auto kPa = units::pascals_t<long double>{1000};
constexpr auto MPa = units::pascals_t<long double>{1e6};
constexpr auto bar = units::bars_t<long double>{1};
constexpr auto atm = units::atmospheres_t<long double>{1};
constexpr auto mmHg = units::millimeters_of_mercury_t<long double>{1};
constexpr auto psi = units::pounds_per_square_inch_t<long double>{1};

// Electric charge units
constexpr auto C = units::coulombs_t<long double>{1};
constexpr auto Ah = units::ampere_hours_t<long double>{1};

// Electric potential units
constexpr auto V = units::volts_t<long double>{1};
constexpr auto mV = units::volts_t<long double>{0.001};
constexpr auto kV = units::volts_t<long double>{1000};

// Capacitance units
constexpr auto F = units::farads_t<long double>{1};
constexpr auto uF = units::farads_t<long double>{1e-6};
constexpr auto nF = units::farads_t<long double>{1e-9};
constexpr auto pF = units::farads_t<long double>{1e-12};

// Resistance units
constexpr auto ohm = units::ohms_t<long double>{1};
constexpr auto kohm = units::ohms_t<long double>{1000};
constexpr auto Mohm = units::ohms_t<long double>{1e6};

// Conductance units
constexpr auto S = units::siemens_t<long double>{1};
constexpr auto mS = units::siemens_t<long double>{0.001};
constexpr auto uS = units::siemens_t<long double>{1e-6};

// Magnetic flux units
constexpr auto Wb = units::webers_t<long double>{1};
constexpr auto Mx = units::maxwells_t<long double>{1};

// Magnetic flux density units
constexpr auto T = units::teslas_t<long double>{1};
constexpr auto G = units::gauss_t<long double>{1};

// Inductance units
constexpr auto H = units::henries_t<long double>{1};
constexpr auto mH = units::henries_t<long double>{0.001};
constexpr auto uH = units::henries_t<long double>{1e-6};

// Area units
constexpr auto m2 = units::square_meters_t<long double>{1};
constexpr auto cm2 = units::square_meters_t<long double>{1e-4};
constexpr auto mm2 = units::square_meters_t<long double>{1e-6};
constexpr auto ft2 = units::square_feet_t<long double>{1};
constexpr auto in2 = units::square_inches_t<long double>{1};
constexpr auto km2 = units::square_kilometers_t<long double>{1};
constexpr auto mi2 = units::square_miles_t<long double>{1};
constexpr auto ha = units::hectares_t<long double>{1};
constexpr auto acre = units::acres_t<long double>{1};

// Volume units
constexpr auto m3 = units::cubic_meters_t<long double>{1};
constexpr auto cm3 = units::cubic_meters_t<long double>{1e-6};
constexpr auto mm3 = units::cubic_meters_t<long double>{1e-9};
constexpr auto L = units::liters_t<long double>{1};
constexpr auto mL = units::liters_t<long double>{0.001};
constexpr auto ft3 = units::cubic_feet_t<long double>{1};
constexpr auto in3 = units::cubic_inches_t<long double>{1};
constexpr auto gal = units::gallons_t<long double>{1};

// Density units
constexpr auto kgpm3 = units::kilograms_per_cubic_meter_t<long double>{1};
constexpr auto gpcm3 = units::grams_per_cubic_centimeter_t<long double>{1};
constexpr auto lbpft3 = units::pounds_per_cubic_foot_t<long double>{1};

// Frequency units
constexpr auto Hz = units::hertz_t<long double>{1};
constexpr auto kHz = units::kilohertz_t<long double>{1};
constexpr auto MHz = units::megahertz_t<long double>{1};
constexpr auto GHz = units::gigahertz_t<long double>{1};

// Angular velocity units
constexpr auto radps = units::radians_per_second_t<long double>{1};
constexpr auto degps = units::degrees_per_second_t<long double>{1};
constexpr auto rpm = units::revolutions_per_minute_t<long double>{1};

// Angular acceleration units
constexpr auto radps2 = units::radians_per_second_squared_t<long double>{1};
constexpr auto degps2 = units::degrees_per_second_squared_t<long double>{1};

// Torque units
constexpr auto Nm = units::newton_meters_t<long double>{1};
constexpr auto ftlb = units::foot_pounds_t<long double>{1};

// Moment of inertia units
constexpr auto kgm2 = units::kilogram_square_meters_t<long double>{1};

// Dynamic viscosity units
constexpr auto Pas = units::pascal_seconds_t<long double>{1};
constexpr auto P = units::poise_t<long double>{1};

// Kinematic viscosity units
constexpr auto m2ps = units::square_meters_per_second_t<long double>{1};
constexpr auto St = units::stokes_t<long double>{1};

// Surface tension units
constexpr auto Npm = units::newtons_per_meter_t<long double>{1};
constexpr auto dyncm = units::dynes_per_centimeter_t<long double>{1};

// Specific heat capacity units
constexpr auto JpkgK = units::joules_per_kilogram_kelvin_t<long double>{1};
constexpr auto calpgC = units::calories_per_gram_celsius_t<long double>{1};

// Thermal conductivity units
constexpr auto WpmK = units::watts_per_meter_kelvin_t<long double>{1};

// Electric field strength units
constexpr auto Vpm = units::volts_per_meter_t<long double>{1};

// Permittivity units
constexpr auto Fpm = units::farads_per_meter_t<long double>{1};

// Permeability units
constexpr auto Hpm = units::henries_per_meter_t<long double>{1};

// Absorbed dose units
constexpr auto Gy = units::grays_t<long double>{1};

// Equivalent dose units
constexpr auto Sv = units::sieverts_t<long double>{1};
constexpr auto rem = units::rems_t<long double>{1};

// Catalytic activity units
constexpr auto kat = units::katal_t<long double>{1};
constexpr auto U = units::enzyme_unit_t<long double>{1};

// Concentration units
constexpr auto molpm3 = units::moles_per_cubic_meter_t<long double>{1};
constexpr auto molpL = units::moles_per_liter_t<long double>{1};

// Molality units
constexpr auto molpkg = units::moles_per_kilogram_t<long double>{1};

// Molar mass units
constexpr auto kgpmol = units::kilograms_per_mole_t<long double>{1};
constexpr auto gpmol = units::grams_per_mole_t<long double>{1};

// Luminous flux units
constexpr auto lm = units::lumens_t<long double>{1};

// Illuminance units
constexpr auto lx = units::lux_t<long double>{1};
constexpr auto fc = units::foot_candles_t<long double>{1};

// Luminous energy units
constexpr auto lms = units::lumen_seconds_t<long double>{1};

// Luminous exposure units
constexpr auto lxs = units::lux_seconds_t<long double>{1};

// Radioactivity units
constexpr auto Bq = units::becquerels_t<long double>{1};
constexpr auto Ci = units::curies_t<long double>{1};

// Flow rate units
constexpr auto m3ps = units::cubic_meters_per_second_t<long double>{1};
constexpr auto Lps = units::liters_per_second_t<long double>{1};
constexpr auto galpm = units::gallons_per_minute_t<long double>{1};

} // namespace squint::units::dsl

#endif // SQUINT_QUANTITY_UNIT_DSL_HPP