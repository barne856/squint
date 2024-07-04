#ifndef SQUINT_QUANTITY_HPP
#define SQUINT_QUANTITY_HPP

#include "squint/dimension.hpp"
#include <cmath>
#include <compare>
#include <concepts>
#include <iostream>
#include <numbers>
#include <type_traits>

namespace squint {

namespace detail {
// Constexpr power function for integer exponents
template <typename T> constexpr T int_pow(T base, int exp) {
    if (exp == 0)
        return T(1);
    if (exp < 0) {
        base = T(1) / base;
        exp = -exp;
    }
    T result = T(1);
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}
// Constexpr square root function
template <typename T> constexpr T sqrt_constexpr(T x, T curr, T prev) {
    return curr == prev ? curr : sqrt_constexpr(x, (curr + x / curr) / 2, curr);
}

template <typename T> constexpr T sqrt_constexpr(T x) {
    return x >= 0 && x < std::numeric_limits<T>::infinity() ? sqrt_constexpr(x, x, T(0))
                                                            : std::numeric_limits<T>::quiet_NaN();
}
} // namespace detail

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <arithmetic T, dimensional D> class quantity {
  public:
    using value_type = T;
    using dimension_type = D;

    // Constructors
    constexpr quantity() noexcept : value_(T{}) {}
    constexpr explicit quantity(const T &value) noexcept : value_(value) {}
    constexpr quantity(const quantity &) noexcept = default;
    constexpr quantity(quantity &&) noexcept = default;

    // Assignment operators
    constexpr quantity &operator=(const quantity &) noexcept = default;
    constexpr quantity &operator=(quantity &&) noexcept = default;

    // Conversion constructor for arithmetic types (implicitly converted to dimensionless)
    template <arithmetic U> constexpr quantity(const U &value) noexcept : value_(static_cast<T>(value)) {
        static_assert(std::is_same_v<D, dimensions::dimensionless>,
                      "Implicit conversion from arithmetic type is only allowed for dimensionless quantities");
    }

    // Accessor methods
    [[nodiscard]] constexpr T value() const noexcept { return value_; }
    [[nodiscard]] constexpr const T *operator->() const noexcept { return &value_; }
    [[nodiscard]] constexpr T *operator->() noexcept { return &value_; }
    [[nodiscard]] constexpr const T &operator*() const noexcept { return value_; }
    [[nodiscard]] constexpr T &operator*() noexcept { return value_; }

    // Explicit conversion to value type
    explicit constexpr operator T() const noexcept { return value_; }

    // Arithmetic operators
    constexpr quantity &operator+=(const quantity &rhs) noexcept {
        value_ += rhs.value_;
        return *this;
    }

    constexpr quantity &operator-=(const quantity &rhs) noexcept {
        value_ -= rhs.value_;
        return *this;
    }

    template <arithmetic U> constexpr quantity &operator*=(const U &scalar) noexcept {
        value_ *= scalar;
        return *this;
    }

    template <arithmetic U> constexpr quantity &operator/=(const U &scalar) noexcept {
        value_ /= scalar;
        return *this;
    }

    // Unary negation operator
    constexpr quantity operator-() const noexcept { return quantity(-value_); }

    // Increment and decrement operators
    constexpr quantity &operator++() noexcept {
        ++value_;
        return *this;
    }

    constexpr quantity operator++(int) noexcept {
        quantity temp(*this);
        ++(*this);
        return temp;
    }

    constexpr quantity &operator--() noexcept {
        --value_;
        return *this;
    }

    constexpr quantity operator--(int) noexcept {
        quantity temp(*this);
        --(*this);
        return temp;
    }

    // Three-way comparison operator
    constexpr auto operator<=>(const quantity &rhs) const noexcept { return value_ <=> rhs.value_; }

    // Equality comparison
    constexpr bool operator==(const quantity &rhs) const noexcept { return value_ == rhs.value_; }

    // Unit conversion
    template <template <typename> typename TargetUnit>
    constexpr T as() const {
        if constexpr (std::is_same_v<TargetUnit<T>, quantity<T, D>>) {
            return value_;
        } else {
            return value_ / TargetUnit<T>::conversion_factor();
        }
    }

    // Power method
    template <int N> constexpr auto pow() const {
        using new_dimension = pow_t<D, N>;
        return quantity<T, new_dimension>(detail::int_pow(value_, N));
    }

    // Root method
    template <int N> auto root() const {
        static_assert(N > 0, "Cannot take 0th root");
        using new_dimension = root_t<D, N>;
        // Note: This is not constexpr, as there's no general constexpr nth root
        return quantity<T, new_dimension>(std::pow(value_, T(1) / N));
    }

    // Square root method
    constexpr auto sqrt() const {
        using new_dimension = root_t<D, 2>;
        return quantity<T, new_dimension>(detail::sqrt_constexpr(value_));
    }

  private:
    T value_;
};

// Arithmetic operations
template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator+(const quantity<T, D> &lhs, const quantity<U, D> &rhs) noexcept {
    return quantity<decltype(lhs.value() + rhs.value()), D>(lhs.value() + rhs.value());
}

template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator-(const quantity<T, D> &lhs, const quantity<U, D> &rhs) noexcept {
    return quantity<decltype(lhs.value() - rhs.value()), D>(lhs.value() - rhs.value());
}

template <arithmetic T, arithmetic U, dimensional D1, dimensional D2>
constexpr auto operator*(const quantity<T, D1> &lhs, const quantity<U, D2> &rhs) noexcept {
    return quantity<decltype(lhs.value() * rhs.value()), mult_t<D1, D2>>(lhs.value() * rhs.value());
}

template <arithmetic T, arithmetic U, dimensional D1, dimensional D2>
constexpr auto operator/(const quantity<T, D1> &lhs, const quantity<U, D2> &rhs) noexcept {
    return quantity<decltype(lhs.value() / rhs.value()), div_t<D1, D2>>(lhs.value() / rhs.value());
}

// Scalar multiplication and division
template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator*(const T &scalar, const quantity<U, D> &q) noexcept {
    return quantity<decltype(scalar * q.value()), D>(scalar * q.value());
}

template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator*(const quantity<T, D> &q, const U &scalar) noexcept {
    return scalar * q;
}

template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator/(const quantity<T, D> &q, const U &scalar) noexcept {
    return quantity<decltype(q.value() / scalar), D>(q.value() / scalar);
}

template <arithmetic T, arithmetic U, dimensional D>
constexpr auto operator/(const T &scalar, const quantity<U, D> &q) noexcept {
    return quantity<decltype(scalar / q.value()), inv_t<D>>(scalar / q.value());
}

// Stream operators
template <arithmetic T, dimensional D> std::ostream &operator<<(std::ostream &os, const quantity<T, D> &q) {
    return os << q.value();
}

template <arithmetic T, dimensional D> std::istream &operator>>(std::istream &is, quantity<T, D> &q) {
    T value;
    is >> value;
    q = quantity<T, D>(value);
    return is;
}

// Type trait to check if a type is a quantity
template <typename T> struct is_quantity : std::false_type {};

template <typename T, dimensional D> struct is_quantity<quantity<T, D>> : std::true_type {};

template <typename T> inline constexpr bool is_quantity_v = is_quantity<T>::value;

// Concept for quantities
template <typename T>
concept quantitative = is_quantity_v<T>;


namespace units {

// Base unit type
template <typename T, typename D>
struct unit_t : quantity<T, D> {
    using quantity<T, D>::quantity;
    static constexpr T conversion_factor() { return T(1); }
    // Allow implicit conversion from quantity<T, D>
    constexpr unit_t(const quantity<T, D>& q) : unit_t<T, D>(q.value()) {}
};

// Dimensionless
template <typename T> using dimensionless_t = unit_t<T, dimensions::dimensionless>;

// Length
template <typename T>
struct length_t : unit_t<T, dimensions::length> {
    using unit_t<T, dimensions::length>::unit_t;
    static constexpr length_t<T> meters(T value) { return length_t<T>(value); }
    static constexpr length_t<T> feet(T value) { return length_t<T>(value * T(0.3048)); }
    static constexpr length_t<T> inches(T value) { return length_t<T>(value * T(0.0254)); }
    static constexpr length_t<T> kilometers(T value) { return length_t<T>(value * T(1000.0)); }
    static constexpr length_t<T> miles(T value) { return length_t<T>(value * T(1609.344)); }
};

template <typename T> struct feet_t : length_t<T> {
    using length_t<T>::length_t;
    static constexpr T conversion_factor() { return T(0.3048); }
};

template <typename T> struct inches_t : length_t<T> {
    using length_t<T>::length_t;
    static constexpr T conversion_factor() { return T(0.0254); }
};

template <typename T> struct kilometers_t : length_t<T> {
    using length_t<T>::length_t;
    static constexpr T conversion_factor() { return T(1000.0); }
};

template <typename T> struct miles_t : length_t<T> {
    using length_t<T>::length_t;
    static constexpr T conversion_factor() { return T(1609.344); }
};

// Time
template <typename T>
struct time_t : unit_t<T, dimensions::time> {
    using unit_t<T, dimensions::time>::unit_t;
    static constexpr time_t<T> seconds(T value) { return time_t<T>(value); }
    static constexpr time_t<T> minutes(T value) { return time_t<T>(value * T(60.0)); }
    static constexpr time_t<T> hours(T value) { return time_t<T>(value * T(3600.0)); }
    static constexpr time_t<T> days(T value) { return time_t<T>(value * T(86400.0)); }
};

template <typename T> struct minutes_t : time_t<T> {
    using time_t<T>::time_t;
    static constexpr T conversion_factor() { return T(60.0); }
};

template <typename T> struct hours_t : time_t<T> {
    using time_t<T>::time_t;
    static constexpr T conversion_factor() { return T(3600.0); }
};

template <typename T> struct days_t : time_t<T> {
    using time_t<T>::time_t;
    static constexpr T conversion_factor() { return T(86400.0); }
};

// Mass
template <typename T>
struct mass_t : unit_t<T, dimensions::mass> {
    using unit_t<T, dimensions::mass>::unit_t;
    static constexpr mass_t<T> kilograms(T value) { return mass_t<T>(value); }
    static constexpr mass_t<T> grams(T value) { return mass_t<T>(value * T(0.001)); }
    static constexpr mass_t<T> pounds(T value) { return mass_t<T>(value * T(0.45359237)); }
};

template <typename T> struct grams_t : mass_t<T> {
    using mass_t<T>::mass_t;
    static constexpr T conversion_factor() { return T(0.001); }
};

template <typename T> struct pounds_t : mass_t<T> {
    using mass_t<T>::mass_t;
    static constexpr T conversion_factor() { return T(0.45359237); }
};

// Temperature
template <typename T>
struct temperature_t : unit_t<T, dimensions::temperature> {
    using unit_t<T, dimensions::temperature>::unit_t;
    static constexpr temperature_t<T> kelvin(T value) { return temperature_t<T>(value); }
    static constexpr temperature_t<T> celsius(T value) { return temperature_t<T>(value + T(273.15)); }
    static constexpr temperature_t<T> fahrenheit(T value) { return temperature_t<T>((value - T(32.0)) * T(5.0) / T(9.0) + T(273.15)); }
};

template <typename T> struct celsius_t : temperature_t<T> {
    using temperature_t<T>::temperature_t;
    static constexpr T conversion_factor() { return T(1); }
    static constexpr T offset() { return T(273.15); }
};

template <typename T> struct fahrenheit_t : temperature_t<T> {
    using temperature_t<T>::temperature_t;
    static constexpr T conversion_factor() { return T(5.0) / T(9.0); }
    static constexpr T offset() { return T(273.15) - T(32.0) * T(5.0) / T(9.0); }
};

// Current
template <typename T>
using current_t = unit_t<T, dimensions::current>;

// Amount of substance
template <typename T>
using amount_of_substance_t = unit_t<T, dimensions::amount_of_substance>;

// Luminous intensity
template <typename T>
using luminous_intensity_t = unit_t<T, dimensions::luminous_intensity>;

// Angle
template <typename T>
struct angle_t : unit_t<T, dimensions::dimensionless> {
    using unit_t<T, dimensions::dimensionless>::unit_t;
    static constexpr angle_t<T> radians(T value) { return angle_t<T>(value); }
    static constexpr angle_t<T> degrees(T value) { return angle_t<T>(value * std::numbers::pi_v<T> / T(180.0)); }
};

template <typename T> struct degrees_t : angle_t<T> {
    using angle_t<T>::angle_t;
    static constexpr T conversion_factor() { return std::numbers::pi_v<T> / T(180.0); }
};

// Velocity
template <typename T>
struct velocity_t : unit_t<T, dimensions::velocity> {
    using unit_t<T, dimensions::velocity>::unit_t;
    static constexpr velocity_t<T> meters_per_second(T value) { return velocity_t<T>(value); }
    static constexpr velocity_t<T> kilometers_per_hour(T value) { return velocity_t<T>(value / T(3.6)); }
    static constexpr velocity_t<T> miles_per_hour(T value) { return velocity_t<T>(value * T(0.44704)); }
};

template <typename T> struct kilometers_per_hour_t : velocity_t<T> {
    using velocity_t<T>::velocity_t;
    static constexpr T conversion_factor() { return T(1) / T(3.6); }
};

template <typename T> struct miles_per_hour_t : velocity_t<T> {
    using velocity_t<T>::velocity_t;
    static constexpr T conversion_factor() { return T(0.44704); }
};

// Acceleration
template <typename T>
struct acceleration_t : unit_t<T, dimensions::acceleration> {
    using unit_t<T, dimensions::acceleration>::unit_t;
    static constexpr acceleration_t<T> meters_per_second_squared(T value) { return acceleration_t<T>(value); }
};

// Area
template <typename T>
struct area_t : unit_t<T, dimensions::area> {
    using unit_t<T, dimensions::area>::unit_t;
    static constexpr area_t<T> square_meters(T value) { return area_t<T>(value); }
    static constexpr area_t<T> square_feet(T value) { return area_t<T>(value * T(0.09290304)); }
    static constexpr area_t<T> acres(T value) { return area_t<T>(value * T(4046.8564224)); }
};

template <typename T> struct square_feet_t : area_t<T> {
    using area_t<T>::area_t;
    static constexpr T conversion_factor() { return T(0.09290304); }
};

template <typename T> struct acres_t : area_t<T> {
    using area_t<T>::area_t;
    static constexpr T conversion_factor() { return T(4046.8564224); }
};

// Volume
template <typename T>
struct volume_t : unit_t<T, dimensions::volume> {
    using unit_t<T, dimensions::volume>::unit_t;
    static constexpr volume_t<T> cubic_meters(T value) { return volume_t<T>(value); }
    static constexpr volume_t<T> liters(T value) { return volume_t<T>(value * T(0.001)); }
    static constexpr volume_t<T> gallons(T value) { return volume_t<T>(value * T(0.00378541)); }
};

template <typename T> struct liters_t : volume_t<T> {
    using volume_t<T>::volume_t;
    static constexpr T conversion_factor() { return T(0.001); }
};

template <typename T> struct gallons_t : volume_t<T> {
    using volume_t<T>::volume_t;
    static constexpr T conversion_factor() { return T(0.00378541); }
};

// Force
template <typename T>
struct force_t : unit_t<T, dimensions::force> {
    using unit_t<T, dimensions::force>::unit_t;
    static constexpr force_t<T> newtons(T value) { return force_t<T>(value); }
    static constexpr force_t<T> pounds_force(T value) { return force_t<T>(value * T(4.448222)); }
};

template <typename T> struct pounds_force_t : force_t<T> {
    using force_t<T>::force_t;
    static constexpr T conversion_factor() { return T(4.448222); }
};

// Pressure
template <typename T>
struct pressure_t : unit_t<T, dimensions::pressure> {
    using unit_t<T, dimensions::pressure>::unit_t;
    static constexpr pressure_t<T> pascals(T value) { return pressure_t<T>(value); }
    static constexpr pressure_t<T> bars(T value) { return pressure_t<T>(value * T(100000.0)); }
    static constexpr pressure_t<T> psi(T value) { return pressure_t<T>(value * T(6894.75729)); }
};

template <typename T> struct bars_t : pressure_t<T> {
    using pressure_t<T>::pressure_t;
    static constexpr T conversion_factor() { return T(100000.0); }
};

template <typename T> struct psi_t : pressure_t<T> {
    using pressure_t<T>::pressure_t;
    static constexpr T conversion_factor() { return T(6894.75729); }
};

// Energy
template <typename T>
struct energy_t : unit_t<T, dimensions::energy> {
    using unit_t<T, dimensions::energy>::unit_t;
    static constexpr energy_t<T> joules(T value) { return energy_t<T>(value); }
    static constexpr energy_t<T> kilowatt_hours(T value) { return energy_t<T>(value * T(3600000.0)); }
};

template <typename T> struct kilowatt_hours_t : energy_t<T> {
    using energy_t<T>::energy_t;
    static constexpr T conversion_factor() { return T(3600000.0); }
};

// Power
template <typename T>
struct power_t : unit_t<T, dimensions::power> {
    using unit_t<T, dimensions::power>::unit_t;
    static constexpr power_t<T> watts(T value) { return power_t<T>(value); }
    static constexpr power_t<T> horsepower(T value) { return power_t<T>(value * T(745.699872)); }
};

template <typename T> struct horsepower_t : power_t<T> {
    using power_t<T>::power_t;
    static constexpr T conversion_factor() { return T(745.699872); }
};

// Other derived units
template <typename T> using density_t = unit_t<T, dimensions::density>;
template <typename T> using charge_t = unit_t<T, dimensions::charge>;
template <typename T> using voltage_t = unit_t<T, dimensions::voltage>;
template <typename T> using capacitance_t = unit_t<T, dimensions::capacitance>;
template <typename T> using resistance_t = unit_t<T, dimensions::resistance>;
template <typename T> using conductance_t = unit_t<T, dimensions::conductance>;
template <typename T> using magnetic_flux_t = unit_t<T, dimensions::magnetic_flux>;
template <typename T> using magnetic_flux_density_t = unit_t<T, dimensions::magnetic_flux_density>;
template <typename T> using inductance_t = unit_t<T, dimensions::inductance>;
template <typename T> using frequency_t = unit_t<T, dimensions::frequency>;
template <typename T> using angular_velocity_t = unit_t<T, dimensions::angular_velocity>;
template <typename T> using momentum_t = unit_t<T, dimensions::momentum>;
template <typename T> using angular_momentum_t = unit_t<T, dimensions::angular_momentum>;
template <typename T> using torque_t = unit_t<T, dimensions::torque>;
template <typename T> using surface_tension_t = unit_t<T, dimensions::surface_tension>;
template <typename T> using dynamic_viscosity_t = unit_t<T, dimensions::dynamic_viscosity>;
template <typename T> using kinematic_viscosity_t = unit_t<T, dimensions::kinematic_viscosity>;
template <typename T> using heat_capacity_t = unit_t<T, dimensions::heat_capacity>;
template <typename T> using specific_heat_capacity_t = unit_t<T, dimensions::specific_heat_capacity>;
template <typename T> using thermal_conductivity_t = unit_t<T, dimensions::thermal_conductivity>;
template <typename T> using electric_field_strength_t = unit_t<T, dimensions::electric_field_strength>;
template <typename T> using electric_displacement_t = unit_t<T, dimensions::electric_displacement>;
template <typename T> using permittivity_t = unit_t<T, dimensions::permittivity>;
template <typename T> using permeability_t = unit_t<T, dimensions::permeability>;
template <typename T> using molar_energy_t = unit_t<T, dimensions::molar_energy>;
template <typename T> using molar_entropy_t = unit_t<T, dimensions::molar_entropy>;
template <typename T> using exposure_t = unit_t<T, dimensions::exposure>;
template <typename T> using dose_equivalent_t = unit_t<T, dimensions::dose_equivalent>;
template <typename T> using catalytic_activity_t = unit_t<T, dimensions::catalytic_activity>;
template <typename T> using wave_number_t = unit_t<T, dimensions::wave_number>;

// Convenience typedefs for float types
// TODO add for double precision units?
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

// TODO add namespace constants_d for double precision constants?
namespace constants {

// Mathematical constants
inline constexpr auto pi = units::dimensionless(std::numbers::pi_v<float>);
inline constexpr auto e = units::dimensionless(std::numbers::e_v<float>);
inline constexpr auto sqrt2 = units::dimensionless(std::numbers::sqrt2_v<float>);
inline constexpr auto ln2 = units::dimensionless(std::numbers::ln2_v<float>);
inline constexpr auto phi = units::dimensionless(1.618033988749895f); // Golden ratio

// Physical constants
namespace si {
// Speed of light in vacuum
inline constexpr auto c = units::velocity::meters_per_second(299'792'458.0f);

// Planck constant
inline constexpr auto h = units::energy::joules(6.62607015e-34f) * units::time::seconds(1.0f);

// Reduced Planck constant (h-bar)
inline constexpr auto hbar = h / (2 * pi);

// Gravitational constant
inline constexpr auto G =
    units::force::newtons(6.67430e-11f) * units::area::square_meters(1.0f) / units::mass::kilograms(1.0f).pow<2>();

// Elementary charge
inline constexpr auto e_charge = units::charge(1.602176634e-19f);

// Electron mass
inline constexpr auto m_e = units::mass::kilograms(9.1093837015e-31f);

// Proton mass
inline constexpr auto m_p = units::mass::kilograms(1.67262192369e-27f);

// Fine-structure constant
inline constexpr auto alpha = units::dimensionless(7.2973525693e-3f);

// Boltzmann constant
inline constexpr auto k_B = units::energy::joules(1.380649e-23f) / units::temperature(1.0f);

// Avogadro constant
inline constexpr auto N_A = units::dimensionless(6.02214076e23f) / units::amount_of_substance(1.0f);

// Gas constant
inline constexpr auto R = k_B * N_A;

// Vacuum electric permittivity
inline constexpr auto epsilon_0 = units::capacitance(8.8541878128e-12f) / units::length::meters(1.0f);

// Vacuum magnetic permeability
inline constexpr auto mu_0 = units::inductance(1.25663706212e-6f) / units::length::meters(1.0f);

// Stefan-Boltzmann constant
inline constexpr auto sigma =
    units::power::watts(5.670374419e-8f) / (units::area::square_meters(1.0f) * units::temperature(1.0f).pow<4>());
} // namespace si

// Astronomical constants
namespace astro {
// Astronomical Unit
inline constexpr auto AU = units::length::meters(1.495978707e11f);

// Parsec
inline constexpr auto parsec = units::length::meters(3.0856775814913673e16f);

// Light year
inline constexpr auto light_year = si::c * units::time::seconds(365.25f * 24 * 3600);

// Solar mass
inline constexpr auto solar_mass = units::mass::kilograms(1.988847e30f);

// Earth mass
inline constexpr auto earth_mass = units::mass::kilograms(5.97217e24f);

// Earth radius (equatorial)
inline constexpr auto earth_radius = units::length::meters(6.3781e6f);

// Standard gravitational acceleration on Earth
inline constexpr auto g = units::acceleration(9.80665f);
} // namespace astro

// Atomic and nuclear constants
namespace atomic {
// Rydberg constant
inline constexpr auto R_inf = units::wave_number(10973731.568160f);

// Bohr radius
inline constexpr auto a_0 = units::length::meters(5.29177210903e-11f);

// Classical electron radius
inline constexpr auto r_e = units::length::meters(2.8179403262e-15f);

// Proton-electron mass ratio
inline constexpr auto m_p_m_e = si::m_p / si::m_e;
} // namespace atomic

} // namespace constants

} // namespace squint

#endif // SQUINT_QUANTITY_HPP