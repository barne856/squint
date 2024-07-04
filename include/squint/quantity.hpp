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

    // unit conversion
    template <typename TargetUnit> constexpr T as() const {
        if constexpr (std::is_same_v<TargetUnit, quantity<T, D>>) {
            return value_;
        } else {
            return value_ / TargetUnit::conversion_factor;
        }
    }

    // Power method
    template <int N> constexpr auto pow() const {
        using new_dimension = pow_t<D, N>;
        return quantity<T, new_dimension>(detail::int_pow(value_, N));
    }

    // Root method
    template <int N> auto root() const {
        static_assert(N != 0, "Cannot take 0th root");
        using new_dimension = root_t<D, N>;
        // Note: This is not constexpr, as there's no general constexpr nth root
        return quantity<T, new_dimension>(std::pow(value_, T(1) / N));
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

// TODO add namespace units_d for double precision units
namespace units {

// Base units
using dimensionless = quantity<float, dimensions::dimensionless>;

namespace length {
struct meters : quantity<float, dimensions::length> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct feet : quantity<float, dimensions::length> {
    constexpr feet(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.3048f;
};
struct inches : quantity<float, dimensions::length> {
    constexpr inches(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.0254f;
};
struct kilometers : quantity<float, dimensions::length> {
    constexpr kilometers(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 1000.0f;
};
struct miles : quantity<float, dimensions::length> {
    constexpr miles(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 1609.344f;
};
} // namespace length

namespace time {
struct seconds : quantity<float, dimensions::time> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct minutes : quantity<float, dimensions::time> {
    constexpr minutes(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 60.0f;
};
struct hours : quantity<float, dimensions::time> {
    constexpr hours(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 3600.0f;
};
struct days : quantity<float, dimensions::time> {
    constexpr days(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 86400.0f;
};
} // namespace time

namespace mass {
struct kilograms : quantity<float, dimensions::mass> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct grams : quantity<float, dimensions::mass> {
    constexpr grams(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.001f;
};
struct pounds : quantity<float, dimensions::mass> {
    constexpr pounds(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.45359237f;
};
} // namespace mass

struct temperature : quantity<float, dimensions::temperature> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct current : quantity<float, dimensions::current> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct amount_of_substance : quantity<float, dimensions::amount_of_substance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct luminous_intensity : quantity<float, dimensions::luminous_intensity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

// Other dimensionless quantities
namespace angle {
struct radians : dimensionless {
    using dimensionless::dimensionless;
    static constexpr float conversion_factor = 1.0f;
};
struct degrees : dimensionless {
    constexpr degrees(float value) : dimensionless(value * conversion_factor) {}
    static constexpr float conversion_factor = std::numbers::pi_v<float> / 180.0f;
};
} // namespace angle

using solid_angle = dimensionless;
using strain = dimensionless;
using refractive_index = dimensionless;

// Derived units
namespace velocity {
struct meters_per_second : quantity<float, dimensions::velocity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct kilometers_per_hour : quantity<float, dimensions::velocity> {
    constexpr kilometers_per_hour(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 1.0f / 3.6f;
};
struct miles_per_hour : quantity<float, dimensions::velocity> {
    constexpr miles_per_hour(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.44704f;
};
} // namespace velocity

struct acceleration : quantity<float, dimensions::acceleration> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

namespace area {
struct square_meters : quantity<float, dimensions::area> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct square_feet : quantity<float, dimensions::area> {
    constexpr square_feet(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.09290304f;
};
struct acres : quantity<float, dimensions::area> {
    constexpr acres(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 4046.8564224f;
};
} // namespace area

namespace volume {
struct cubic_meters : quantity<float, dimensions::volume> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct liters : quantity<float, dimensions::volume> {
    constexpr liters(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.001f;
};
struct gallons : quantity<float, dimensions::volume> {
    constexpr gallons(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 0.00378541f;
};
} // namespace volume

struct density : quantity<float, dimensions::density> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

namespace force {
struct newtons : quantity<float, dimensions::force> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct pounds_force : quantity<float, dimensions::force> {
    constexpr pounds_force(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 4.448222f;
};
} // namespace force

struct force_density : quantity<float, dimensions::force_density> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

namespace pressure {
struct pascals : quantity<float, dimensions::pressure> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct bars : quantity<float, dimensions::pressure> {
    constexpr bars(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 100000.0f;
};
struct psi : quantity<float, dimensions::pressure> {
    constexpr psi(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 6894.75729f;
};
} // namespace pressure

struct dynamic_viscosity : quantity<float, dimensions::dynamic_viscosity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct kinematic_viscosity : quantity<float, dimensions::kinematic_viscosity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct flow : quantity<float, dimensions::flow> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

namespace energy {
struct joules : quantity<float, dimensions::energy> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct kilowatt_hours : quantity<float, dimensions::energy> {
    constexpr kilowatt_hours(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 3600000.0f;
};
} // namespace energy

namespace power {
struct watts : quantity<float, dimensions::power> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};
struct horsepower : quantity<float, dimensions::power> {
    constexpr horsepower(float value) : quantity(value * conversion_factor) {}
    static constexpr float conversion_factor = 745.699872f;
};
} // namespace power

struct charge : quantity<float, dimensions::charge> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct voltage : quantity<float, dimensions::voltage> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct capacitance : quantity<float, dimensions::capacitance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct resistance : quantity<float, dimensions::resistance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct conductance : quantity<float, dimensions::conductance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct magnetic_flux : quantity<float, dimensions::magnetic_flux> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct magnetic_flux_density : quantity<float, dimensions::magnetic_flux_density> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct inductance : quantity<float, dimensions::inductance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct frequency : quantity<float, dimensions::frequency> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct angular_velocity : quantity<float, dimensions::angular_velocity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct momentum : quantity<float, dimensions::momentum> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct angular_momentum : quantity<float, dimensions::angular_momentum> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct torque : quantity<float, dimensions::torque> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct surface_tension : quantity<float, dimensions::surface_tension> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct heat_capacity : quantity<float, dimensions::heat_capacity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct specific_heat_capacity : quantity<float, dimensions::specific_heat_capacity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct thermal_conductivity : quantity<float, dimensions::thermal_conductivity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct electric_field_strength : quantity<float, dimensions::electric_field_strength> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct electric_displacement : quantity<float, dimensions::electric_displacement> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct permittivity : quantity<float, dimensions::permittivity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct permeability : quantity<float, dimensions::permeability> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct molar_energy : quantity<float, dimensions::molar_energy> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct molar_entropy : quantity<float, dimensions::molar_entropy> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct exposure : quantity<float, dimensions::exposure> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct dose_equivalent : quantity<float, dimensions::dose_equivalent> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct catalytic_activity : quantity<float, dimensions::catalytic_activity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct luminance : quantity<float, dimensions::luminance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct magnetic_field_strength : quantity<float, dimensions::magnetic_field_strength> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct molarity : quantity<float, dimensions::molarity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct molar_mass : quantity<float, dimensions::molar_mass> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct impulse : quantity<float, dimensions::impulse> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct wave_number : quantity<float, dimensions::wave_number> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct specific_volume : quantity<float, dimensions::specific_volume> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct radiant_intensity : quantity<float, dimensions::radiant_intensity> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct radiance : quantity<float, dimensions::radiance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct irradiance : quantity<float, dimensions::irradiance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

struct thermal_resistance : quantity<float, dimensions::thermal_resistance> {
    using quantity::quantity;
    static constexpr float conversion_factor = 1.0f;
};

} // namespace units

// TODO add namespace constants_d for double precision constants
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