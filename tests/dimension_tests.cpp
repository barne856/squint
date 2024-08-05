// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/dimension.hpp"
#include "squint/quantity/dimension_types.hpp"

TEST_CASE("Base dimensions are correctly defined") {
    using namespace squint::dimensions;

    static_assert(std::is_same_v<dimensionless::L, std::ratio<0>>);
    static_assert(std::is_same_v<length::L, std::ratio<1>>);
    static_assert(std::is_same_v<time::T, std::ratio<1>>);
    static_assert(std::is_same_v<mass::M, std::ratio<1>>);
    static_assert(std::is_same_v<temperature::K, std::ratio<1>>);
    static_assert(std::is_same_v<current::I, std::ratio<1>>);
    static_assert(std::is_same_v<amount_of_substance::N, std::ratio<1>>);
    static_assert(std::is_same_v<luminous_intensity::J, std::ratio<1>>);
}

TEST_CASE("Dimension arithmetic operations work correctly") {
    using namespace squint;
    using namespace squint::dimensions;

    SUBCASE("Multiplication") {
        using velocity = squint::div_t<length, squint::dimensions::time>;
        using acceleration = squint::div_t<velocity, squint::dimensions::time>;
        static_assert(std::is_same_v<mult_t<velocity, squint::dimensions::time>, length>);
        static_assert(std::is_same_v<mult_t<acceleration, squint::dimensions::time>, velocity>);
    }

    SUBCASE("Division") {
        using velocity = squint::div_t<length, squint::dimensions::time>;
        static_assert(std::is_same_v<squint::div_t<length, velocity>, squint::dimensions::time>);
    }

    SUBCASE("Power") {
        static_assert(std::is_same_v<pow_t<length, 2>, area>);
        static_assert(std::is_same_v<pow_t<length, 3>, volume>);
    }

    SUBCASE("Root") {
        static_assert(std::is_same_v<root_t<area, 2>, length>);
        static_assert(std::is_same_v<root_t<volume, 3>, length>);
    }

    SUBCASE("Inversion") {
        using frequency = inv_t<squint::dimensions::time>;
        static_assert(std::is_same_v<mult_t<frequency, squint::dimensions::time>, squint::dimensions::dimensionless>);
    }
}

TEST_CASE("Derived dimensions are correctly defined") {
    using namespace squint::dimensions;

    SUBCASE("Mechanical dimensions") {
        static_assert(std::is_same_v<velocity, squint::div_t<length, squint::dimensions::time>>);
        static_assert(std::is_same_v<acceleration, squint::div_t<velocity, squint::dimensions::time>>);
        static_assert(std::is_same_v<force, squint::mult_t<mass, acceleration>>);
        static_assert(std::is_same_v<energy, squint::mult_t<force, length>>);
        static_assert(std::is_same_v<power, squint::div_t<energy, squint::dimensions::time>>);
    }

    SUBCASE("Electromagnetic dimensions") {
        static_assert(std::is_same_v<charge, squint::mult_t<current, squint::dimensions::time>>);
        static_assert(std::is_same_v<voltage, squint::div_t<energy, charge>>);
        static_assert(std::is_same_v<capacitance, squint::div_t<charge, voltage>>);
        static_assert(std::is_same_v<resistance, squint::div_t<voltage, current>>);
        static_assert(std::is_same_v<magnetic_flux, squint::mult_t<voltage, squint::dimensions::time>>);
    }

    SUBCASE("Thermodynamic dimensions") {
        static_assert(std::is_same_v<heat_capacity, squint::div_t<energy, temperature>>);
        static_assert(std::is_same_v<specific_heat_capacity, squint::div_t<heat_capacity, mass>>);
        static_assert(std::is_same_v<thermal_conductivity, squint::div_t<power, squint::mult_t<length, temperature>>>);
    }

    SUBCASE("Chemical dimensions") {
        static_assert(std::is_same_v<molarity, squint::div_t<amount_of_substance, volume>>);
        static_assert(std::is_same_v<molar_mass, squint::div_t<mass, amount_of_substance>>);
        static_assert(std::is_same_v<catalytic_activity, squint::div_t<amount_of_substance, squint::dimensions::time>>);
    }

    SUBCASE("Optical dimensions") {
        static_assert(std::is_same_v<luminance, squint::div_t<luminous_intensity, area>>);
        static_assert(std::is_same_v<radiant_intensity, squint::div_t<power, solid_angle>>);
        static_assert(std::is_same_v<radiance, squint::div_t<radiant_intensity, area>>);
        static_assert(std::is_same_v<irradiance, squint::div_t<power, area>>);
    }
}

TEST_CASE("Complex dimension combinations") {
    using namespace squint;
    using namespace squint::dimensions;

    SUBCASE("Pressure * Volume = Energy") {
        using pressure_volume = mult_t<pressure, volume>;
        static_assert(std::is_same_v<pressure_volume, energy>);
    }

    SUBCASE("Force * Time = Momentum") {
        using force_time = mult_t<force, squint::dimensions::time>;
        static_assert(std::is_same_v<force_time, momentum>);
    }

    SUBCASE("Power / Velocity = Force") {
        using power_over_velocity = squint::div_t<power, velocity>;
        static_assert(std::is_same_v<power_over_velocity, force>);
    }

    SUBCASE("Energy / (Mass * Temperature) = Specific Heat Capacity") {
        using energy_over_mass_temp = squint::div_t<energy, mult_t<mass, temperature>>;
        static_assert(std::is_same_v<energy_over_mass_temp, specific_heat_capacity>);
    }
}

TEST_CASE("Dimensionless quantities") {
    using namespace squint::dimensions;

    static_assert(std::is_same_v<angle, dimensionless>);
    static_assert(std::is_same_v<solid_angle, dimensionless>);
    static_assert(std::is_same_v<strain, dimensionless>);
    static_assert(std::is_same_v<refractive_index, dimensionless>);
}

TEST_CASE("Dimension arithmetic with fractional exponents") {
    using namespace squint;
    using namespace squint::dimensions;

    SUBCASE("Square root of area is length") { static_assert(std::is_same_v<root_t<area, 2>, length>); }

    SUBCASE("Cube root of volume is length") { static_assert(std::is_same_v<root_t<volume, 3>, length>); }

    SUBCASE("Square root of energy per mass is velocity") {
        using energy_per_mass = squint::div_t<energy, mass>;
        static_assert(std::is_same_v<root_t<energy_per_mass, 2>, velocity>);
    }
}

TEST_CASE("Dimension arithmetic error cases") {
    using namespace squint;
    using namespace squint::dimensions;

    SUBCASE("Cannot take 0th root") {
        // This should cause a compile-time error
        // Uncomment to test:
        // using invalid_root = root_t<length, 0>;
    }
}
// NOLINTEND