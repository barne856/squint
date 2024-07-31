#ifndef SQUINT_QUANTITY_UNITS_HPP
#define SQUINT_QUANTITY_UNITS_HPP

#include "squint/quantity/quantity.hpp"

namespace squint {
namespace units {

// Base unit type
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
struct unit_t : quantity<T, D, ErrorChecking> {
    using quantity<T, D, ErrorChecking>::quantity;
    static constexpr T convert_to(const unit_t &u, const unit_t & /*unused*/) { return u.value(); }
    // Allow implicit conversion from quantity<T, D, ErrorChecking>
    constexpr unit_t(const quantity<T, D, ErrorChecking> &q) : unit_t<T, D, ErrorChecking>(q.value()) {}
};

// Dimensionless
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dimensionless_t = unit_t<T, dimensions::dimensionless, ErrorChecking>;

// Length
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct length_t : unit_t<T, dimensions::length, ErrorChecking> {
    using unit_t<T, dimensions::length, ErrorChecking>::unit_t;
    static constexpr length_t<T, ErrorChecking> meters(T value) { return length_t<T, ErrorChecking>(value); }
    static constexpr length_t<T, ErrorChecking> feet(T value) { return length_t<T, ErrorChecking>(value * T(0.3048)); }
    static constexpr length_t<T, ErrorChecking> inches(T value) {
        return length_t<T, ErrorChecking>(value * T(0.0254));
    }
    static constexpr length_t<T, ErrorChecking> kilometers(T value) {
        return length_t<T, ErrorChecking>(value * T(1000.0));
    }
    static constexpr length_t<T, ErrorChecking> miles(T value) {
        return length_t<T, ErrorChecking>(value * T(1609.344));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct feet_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const feet_t & /*unused*/) {
        return l.value() / T(0.3048);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct inches_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const inches_t & /*unused*/) {
        return l.value() / T(0.0254);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilometers_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const kilometers_t & /*unused*/) {
        return l.value() / T(1000.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct miles_t : length_t<T, ErrorChecking> {
    using length_t<T, ErrorChecking>::length_t;
    static constexpr T convert_to(const length_t<T, ErrorChecking> &l, const miles_t & /*unused*/) {
        return l.value() / T(1609.344);
    }
};

// Time
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct time_t : unit_t<T, dimensions::time, ErrorChecking> {
    using unit_t<T, dimensions::time, ErrorChecking>::unit_t;
    static constexpr time_t<T, ErrorChecking> seconds(T value) { return time_t<T, ErrorChecking>(value); }
    static constexpr time_t<T, ErrorChecking> minutes(T value) { return time_t<T, ErrorChecking>(value * T(60.0)); }
    static constexpr time_t<T, ErrorChecking> hours(T value) { return time_t<T, ErrorChecking>(value * T(3600.0)); }
    static constexpr time_t<T, ErrorChecking> days(T value) { return time_t<T, ErrorChecking>(value * T(86400.0)); }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct minutes_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const minutes_t & /*unused*/) {
        return t.value() / T(60.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct hours_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const hours_t & /*unused*/) {
        return t.value() / T(3600.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct days_t : time_t<T, ErrorChecking> {
    using time_t<T, ErrorChecking>::time_t;
    static constexpr T convert_to(const time_t<T, ErrorChecking> &t, const days_t & /*unused*/) {
        return t.value() / T(86400.0);
    }
};

// Mass
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct mass_t : unit_t<T, dimensions::mass, ErrorChecking> {
    using unit_t<T, dimensions::mass, ErrorChecking>::unit_t;
    static constexpr mass_t<T, ErrorChecking> kilograms(T value) { return mass_t<T, ErrorChecking>(value); }
    static constexpr mass_t<T, ErrorChecking> grams(T value) { return mass_t<T, ErrorChecking>(value * T(0.001)); }
    static constexpr mass_t<T, ErrorChecking> pounds(T value) {
        return mass_t<T, ErrorChecking>(value * T(0.45359237));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct grams_t : mass_t<T, ErrorChecking> {
    using mass_t<T, ErrorChecking>::mass_t;
    static constexpr T convert_to(const mass_t<T, ErrorChecking> &m, const grams_t & /*unused*/) {
        return m.value() / T(0.001);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pounds_t : mass_t<T, ErrorChecking> {
    using mass_t<T, ErrorChecking>::mass_t;
    static constexpr T convert_to(const mass_t<T, ErrorChecking> &m, const pounds_t & /*unused*/) {
        return m.value() / T(0.45359237);
    }
};

// Temperature
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct temperature_t : unit_t<T, dimensions::temperature, ErrorChecking> {
    using unit_t<T, dimensions::temperature, ErrorChecking>::unit_t;
    static constexpr temperature_t<T, ErrorChecking> kelvin(T value) { return temperature_t<T, ErrorChecking>(value); }
    static constexpr temperature_t<T, ErrorChecking> celsius(T value) {
        return temperature_t<T, ErrorChecking>(value + T(273.15));
    }
    static constexpr temperature_t<T, ErrorChecking> fahrenheit(T value) {
        return temperature_t<T, ErrorChecking>((value - T(32.0)) * T(5.0) / T(9.0) + T(273.15));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct celsius_t : temperature_t<T, ErrorChecking> {
    using temperature_t<T, ErrorChecking>::temperature_t;
    static constexpr T convert_to(const temperature_t<T, ErrorChecking> &t, const celsius_t & /*unused*/) {
        return t.value() - T(273.15);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct fahrenheit_t : temperature_t<T, ErrorChecking> {
    using temperature_t<T, ErrorChecking>::temperature_t;
    static constexpr T convert_to(const temperature_t<T, ErrorChecking> &t, const fahrenheit_t & /*unused*/) {
        return t.value() * T(9.0) / T(5.0) - T(459.67);
    }
};

// Current
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using current_t = unit_t<T, dimensions::current, ErrorChecking>;

// Amount of substance
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using amount_of_substance_t = unit_t<T, dimensions::amount_of_substance, ErrorChecking>;

// Luminous intensity
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using luminous_intensity_t = unit_t<T, dimensions::luminous_intensity, ErrorChecking>;

// Angle
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct angle_t : unit_t<T, dimensions::dimensionless, ErrorChecking> {
    using unit_t<T, dimensions::dimensionless, ErrorChecking>::unit_t;
    static constexpr angle_t<T, ErrorChecking> radians(T value) { return angle_t<T, ErrorChecking>(value); }
    static constexpr angle_t<T, ErrorChecking> degrees(T value) {
        return angle_t<T, ErrorChecking>(value * std::numbers::pi_v<T> / T(180.0));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct degrees_t : angle_t<T, ErrorChecking> {
    using angle_t<T, ErrorChecking>::angle_t;
    static constexpr T convert_to(const angle_t<T, ErrorChecking> &a, const degrees_t & /*unused*/) {
        return a.value() * T(180.0) / std::numbers::pi_v<T>;
    }
};

// Velocity
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct velocity_t : unit_t<T, dimensions::velocity, ErrorChecking> {
    using unit_t<T, dimensions::velocity, ErrorChecking>::unit_t;
    static constexpr velocity_t<T, ErrorChecking> meters_per_second(T value) {
        return velocity_t<T, ErrorChecking>(value);
    }
    static constexpr velocity_t<T, ErrorChecking> kilometers_per_hour(T value) {
        return velocity_t<T, ErrorChecking>(value / T(3.6));
    }
    static constexpr velocity_t<T, ErrorChecking> miles_per_hour(T value) {
        return velocity_t<T, ErrorChecking>(value * T(0.44704));
    }
    static constexpr velocity_t<T, ErrorChecking> feet_per_second(T value) {
        return velocity_t<T, ErrorChecking>(value * T(3.28084));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilometers_per_hour_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const kilometers_per_hour_t & /*unused*/) {
        return v.value() * T(3.6);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct miles_per_hour_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const miles_per_hour_t & /*unused*/) {
        return v.value() / T(0.44704);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct feet_per_second_t : velocity_t<T, ErrorChecking> {
    using velocity_t<T, ErrorChecking>::velocity_t;
    static constexpr T convert_to(const velocity_t<T, ErrorChecking> &v, const feet_per_second_t & /*unused*/) {
        return v.value() / T(3.28084);
    }
};

// Acceleration
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct acceleration_t : unit_t<T, dimensions::acceleration, ErrorChecking> {
    using unit_t<T, dimensions::acceleration, ErrorChecking>::unit_t;
    static constexpr acceleration_t<T, ErrorChecking> meters_per_second_squared(T value) {
        return acceleration_t<T, ErrorChecking>(value);
    }
};

// Area
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct area_t : unit_t<T, dimensions::area, ErrorChecking> {
    using unit_t<T, dimensions::area, ErrorChecking>::unit_t;
    static constexpr area_t<T, ErrorChecking> square_meters(T value) { return area_t<T, ErrorChecking>(value); }
    static constexpr area_t<T, ErrorChecking> square_feet(T value) {
        return area_t<T, ErrorChecking>(value * T(0.09290304));
    }
    static constexpr area_t<T, ErrorChecking> acres(T value) {
        return area_t<T, ErrorChecking>(value * T(4046.8564224));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct square_feet_t : area_t<T, ErrorChecking> {
    using area_t<T, ErrorChecking>::area_t;
    static constexpr T convert_to(const area_t<T, ErrorChecking> &a, const square_feet_t & /*unused*/) {
        return a.value() / T(0.09290304);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct acres_t : area_t<T, ErrorChecking> {
    using area_t<T, ErrorChecking>::area_t;
    static constexpr T convert_to(const area_t<T, ErrorChecking> &a, const acres_t & /*unused*/) {
        return a.value() / T(4046.8564224);
    }
};

// Volume
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct volume_t : unit_t<T, dimensions::volume, ErrorChecking> {
    using unit_t<T, dimensions::volume, ErrorChecking>::unit_t;
    static constexpr volume_t<T, ErrorChecking> cubic_meters(T value) { return volume_t<T, ErrorChecking>(value); }
    static constexpr volume_t<T, ErrorChecking> liters(T value) { return volume_t<T, ErrorChecking>(value * T(0.001)); }
    static constexpr volume_t<T, ErrorChecking> gallons(T value) {
        return volume_t<T, ErrorChecking>(value * T(0.00378541));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct liters_t : volume_t<T, ErrorChecking> {
    using volume_t<T, ErrorChecking>::volume_t;
    static constexpr T convert_to(const volume_t<T, ErrorChecking> &v, const liters_t & /*unused*/) {
        return v.value() / T(0.001);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct gallons_t : volume_t<T, ErrorChecking> {
    using volume_t<T, ErrorChecking>::volume_t;
    static constexpr T convert_to(const volume_t<T, ErrorChecking> &v, const gallons_t & /*unused*/) {
        return v.value() / T(0.00378541);
    }
};

// Force
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct force_t : unit_t<T, dimensions::force, ErrorChecking> {
    using unit_t<T, dimensions::force, ErrorChecking>::unit_t;
    static constexpr force_t<T, ErrorChecking> newtons(T value) { return force_t<T, ErrorChecking>(value); }
    static constexpr force_t<T, ErrorChecking> pounds_force(T value) {
        return force_t<T, ErrorChecking>(value * T(4.448222));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pounds_force_t : force_t<T, ErrorChecking> {
    using force_t<T, ErrorChecking>::force_t;
    static constexpr T convert_to(const force_t<T, ErrorChecking> &f, const pounds_force_t & /*unused*/) {
        return f.value() / T(4.448222);
    }
};

// Pressure
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct pressure_t : unit_t<T, dimensions::pressure, ErrorChecking> {
    using unit_t<T, dimensions::pressure, ErrorChecking>::unit_t;
    static constexpr pressure_t<T, ErrorChecking> pascals(T value) { return pressure_t<T, ErrorChecking>(value); }
    static constexpr pressure_t<T, ErrorChecking> bars(T value) {
        return pressure_t<T, ErrorChecking>(value * T(100000.0));
    }
    static constexpr pressure_t<T, ErrorChecking> psi(T value) {
        return pressure_t<T, ErrorChecking>(value * T(6894.75729));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct bars_t : pressure_t<T, ErrorChecking> {
    using pressure_t<T, ErrorChecking>::pressure_t;
    static constexpr T convert_to(const pressure_t<T, ErrorChecking> &p, const bars_t & /*unused*/) {
        return p.value() / T(100000.0);
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct psi_t : pressure_t<T, ErrorChecking> {
    using pressure_t<T, ErrorChecking>::pressure_t;
    static constexpr T convert_to(const pressure_t<T, ErrorChecking> &p, const psi_t & /*unused*/) {
        return p.value() / T(6894.75729);
    }
};

// Energy
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct energy_t : unit_t<T, dimensions::energy, ErrorChecking> {
    using unit_t<T, dimensions::energy, ErrorChecking>::unit_t;
    static constexpr energy_t<T, ErrorChecking> joules(T value) { return energy_t<T, ErrorChecking>(value); }
    static constexpr energy_t<T, ErrorChecking> kilowatt_hours(T value) {
        return energy_t<T, ErrorChecking>(value * T(3600000.0));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct kilowatt_hours_t : energy_t<T, ErrorChecking> {
    using energy_t<T, ErrorChecking>::energy_t;
    static constexpr T convert_to(const energy_t<T, ErrorChecking> &e, const kilowatt_hours_t & /*unused*/) {
        return e.value() / T(3600000.0);
    }
};

// Power
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct power_t : unit_t<T, dimensions::power, ErrorChecking> {
    using unit_t<T, dimensions::power, ErrorChecking>::unit_t;
    static constexpr power_t<T, ErrorChecking> watts(T value) { return power_t<T, ErrorChecking>(value); }
    static constexpr power_t<T, ErrorChecking> horsepower(T value) {
        return power_t<T, ErrorChecking>(value * T(745.699872));
    }
};

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct horsepower_t : power_t<T, ErrorChecking> {
    using power_t<T, ErrorChecking>::power_t;
    static constexpr T convert_to(const power_t<T, ErrorChecking> &p, const horsepower_t & /*unused*/) {
        return p.value() / T(745.699872);
    }
};

// Other derived units
template <typename T, error_checking ErrorChecking = error_checking::disabled>
using density_t = unit_t<T, dimensions::density, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using charge_t = unit_t<T, dimensions::charge, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using voltage_t = unit_t<T, dimensions::voltage, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using capacitance_t = unit_t<T, dimensions::capacitance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using resistance_t = unit_t<T, dimensions::resistance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using conductance_t = unit_t<T, dimensions::conductance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using magnetic_flux_t = unit_t<T, dimensions::magnetic_flux, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using magnetic_flux_density_t = unit_t<T, dimensions::magnetic_flux_density, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using inductance_t = unit_t<T, dimensions::inductance, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using frequency_t = unit_t<T, dimensions::frequency, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using angular_velocity_t = unit_t<T, dimensions::angular_velocity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using momentum_t = unit_t<T, dimensions::momentum, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using angular_momentum_t = unit_t<T, dimensions::angular_momentum, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using torque_t = unit_t<T, dimensions::torque, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using surface_tension_t = unit_t<T, dimensions::surface_tension, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dynamic_viscosity_t = unit_t<T, dimensions::dynamic_viscosity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using kinematic_viscosity_t = unit_t<T, dimensions::kinematic_viscosity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using heat_capacity_t = unit_t<T, dimensions::heat_capacity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using specific_heat_capacity_t = unit_t<T, dimensions::specific_heat_capacity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using thermal_conductivity_t = unit_t<T, dimensions::thermal_conductivity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using electric_field_strength_t = unit_t<T, dimensions::electric_field_strength, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using electric_displacement_t = unit_t<T, dimensions::electric_displacement, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using permittivity_t = unit_t<T, dimensions::permittivity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using permeability_t = unit_t<T, dimensions::permeability, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using molar_energy_t = unit_t<T, dimensions::molar_energy, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using molar_entropy_t = unit_t<T, dimensions::molar_entropy, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using exposure_t = unit_t<T, dimensions::exposure, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using dose_equivalent_t = unit_t<T, dimensions::dose_equivalent, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using catalytic_activity_t = unit_t<T, dimensions::catalytic_activity, ErrorChecking>;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
using wave_number_t = unit_t<T, dimensions::wave_number, ErrorChecking>;

// Convenience typedefs for float types with error checking disabled
using dimensionless = dimensionless_t<float>;
using length = length_t<float>;
using time = time_t<float>;
using mass = mass_t<float>;
using temperature = temperature_t<float>;
using current = current_t<float>;
using amount_of_substance = amount_of_substance_t<float>;
using luminous_intensity = luminous_intensity_t<float>;
using angle = angle_t<float>;
using velocity = velocity_t<float>;
using acceleration = acceleration_t<float>;
using area = area_t<float>;
using volume = volume_t<float>;
using force = force_t<float>;
using pressure = pressure_t<float>;
using energy = energy_t<float>;
using power = power_t<float>;
using density = density_t<float>;
using charge = charge_t<float>;
using voltage = voltage_t<float>;
using capacitance = capacitance_t<float>;
using resistance = resistance_t<float>;
using conductance = conductance_t<float>;
using magnetic_flux = magnetic_flux_t<float>;
using magnetic_flux_density = magnetic_flux_density_t<float>;
using inductance = inductance_t<float>;
using frequency = frequency_t<float>;
using angular_velocity = angular_velocity_t<float>;
using momentum = momentum_t<float>;
using angular_momentum = angular_momentum_t<float>;
using torque = torque_t<float>;
using surface_tension = surface_tension_t<float>;
using dynamic_viscosity = dynamic_viscosity_t<float>;
using kinematic_viscosity = kinematic_viscosity_t<float>;
using heat_capacity = heat_capacity_t<float>;
using specific_heat_capacity = specific_heat_capacity_t<float>;
using thermal_conductivity = thermal_conductivity_t<float>;
using electric_field_strength = electric_field_strength_t<float>;
using electric_displacement = electric_displacement_t<float>;
using permittivity = permittivity_t<float>;
using permeability = permeability_t<float>;
using molar_energy = molar_energy_t<float>;
using molar_entropy = molar_entropy_t<float>;
using exposure = exposure_t<float>;
using dose_equivalent = dose_equivalent_t<float>;
using catalytic_activity = catalytic_activity_t<float>;
using wave_number = wave_number_t<float>;

} // namespace units
} // namespace squint

#endif // SQUINT_QUANTITY_UNITS_HPP