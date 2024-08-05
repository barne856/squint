// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/dimension_types.hpp"

using namespace squint::dimensions;
using namespace squint;

using voltage_dim = dim_div_t<energy_dim, charge_dim>;
using capacitance_dim = dim_div_t<charge_dim, voltage_dim>;
using resistance_dim = dim_div_t<voltage_dim, I>;
using magnetic_flux_dim = dim_mult_t<voltage_dim, T>;
using heat_capacity_dim = dim_div_t<energy_dim, K>;
using specific_heat_capacity_dim = dim_div_t<heat_capacity_dim, M>;
using thermal_conductivity_dim = dim_div_t<power_dim, dim_mult_t<L, K>>;
using molarity_dim = dim_div_t<N, volume_dim>;
using molar_mass_dim = dim_div_t<M, N>;
using catalytic_activity_dim = dim_div_t<N, T>;
using luminance_dim = dim_div_t<J, area_dim>;
using radiant_intensity_dim = dim_div_t<power_dim, unity>;
using radiance_dim = dim_div_t<radiant_intensity_dim, area_dim>;
using irradiance_dim = dim_div_t<power_dim, area_dim>;
using pressure_volume_dim = dim_mult_t<pressure_dim, volume_dim>;
using force_time_dim = dim_mult_t<force_dim, T>;
using power_over_velocity_dim = dim_div_t<power_dim, velocity_dim>;
using energy_over_mass_temp_dim = dim_div_t<energy_dim, dim_mult_t<M, K>>;
using energy_per_mass_dim = dim_div_t<energy_dim, M>;

TEST_CASE("Base dimensions are correctly defined") {
    static_assert(std::is_same_v<unity::L, std::ratio<0>>);
    static_assert(std::is_same_v<L::L, std::ratio<1>>);
    static_assert(std::is_same_v<T::T, std::ratio<1>>);
    static_assert(std::is_same_v<M::M, std::ratio<1>>);
    static_assert(std::is_same_v<K::K, std::ratio<1>>);
    static_assert(std::is_same_v<I::I, std::ratio<1>>);
    static_assert(std::is_same_v<N::N, std::ratio<1>>);
    static_assert(std::is_same_v<J::J, std::ratio<1>>);
}

TEST_CASE("Dimension arithmetic operations work correctly") {

    SUBCASE("Multiplication") {
        static_assert(std::is_same_v<dim_mult_t<velocity_dim, T>, L>);
        static_assert(std::is_same_v<dim_mult_t<acceleration_dim, T>, velocity_dim>);
    }

    SUBCASE("Division") { static_assert(std::is_same_v<dim_div_t<L, velocity_dim>, T>); }

    SUBCASE("Power") {
        static_assert(std::is_same_v<dim_pow_t<L, 2>, area_dim>);
        static_assert(std::is_same_v<dim_pow_t<L, 3>, volume_dim>);
    }

    SUBCASE("Root") {
        static_assert(std::is_same_v<dim_root_t<area_dim, 2>, L>);
        static_assert(std::is_same_v<dim_root_t<volume_dim, 3>, L>);
    }

    SUBCASE("Inversion") { static_assert(std::is_same_v<dim_mult_t<frequency_dim, T>, unity>); }
}

TEST_CASE("Derived dimensions are correctly defined") {

    SUBCASE("Mechanical dimensions") {
        static_assert(std::is_same_v<velocity_dim, dim_div_t<L, T>>);
        static_assert(std::is_same_v<acceleration_dim, dim_div_t<velocity_dim, T>>);
        static_assert(std::is_same_v<force_dim, dim_mult_t<M, acceleration_dim>>);
        static_assert(std::is_same_v<energy_dim, dim_mult_t<force_dim, L>>);
        static_assert(std::is_same_v<power_dim, dim_div_t<energy_dim, T>>);
    }

    SUBCASE("Electromagnetic dimensions") {
        static_assert(std::is_same_v<charge_dim, dim_mult_t<I, T>>);
        static_assert(std::is_same_v<voltage_dim, dim_div_t<energy_dim, charge_dim>>);
        static_assert(std::is_same_v<capacitance_dim, dim_div_t<charge_dim, voltage_dim>>);
        static_assert(std::is_same_v<resistance_dim, dim_div_t<voltage_dim, I>>);
        static_assert(std::is_same_v<magnetic_flux_dim, dim_mult_t<voltage_dim, T>>);
    }

    SUBCASE("Thermodynamic dimensions") {
        static_assert(std::is_same_v<heat_capacity_dim, dim_div_t<energy_dim, K>>);
        static_assert(std::is_same_v<specific_heat_capacity_dim, dim_div_t<heat_capacity_dim, M>>);
        static_assert(std::is_same_v<thermal_conductivity_dim, dim_div_t<power_dim, dim_mult_t<L, K>>>);
    }

    SUBCASE("Chemical dimensions") {
        static_assert(std::is_same_v<molarity_dim, dim_div_t<N, volume_dim>>);
        static_assert(std::is_same_v<molar_mass_dim, dim_div_t<M, N>>);
        static_assert(std::is_same_v<catalytic_activity_dim, dim_div_t<N, T>>);
    }

    SUBCASE("Optical dimensions") {
        static_assert(std::is_same_v<luminance_dim, dim_div_t<J, area_dim>>);
        static_assert(std::is_same_v<radiant_intensity_dim, dim_div_t<power_dim, unity>>);
        static_assert(std::is_same_v<radiance_dim, dim_div_t<radiant_intensity_dim, area_dim>>);
        static_assert(std::is_same_v<irradiance_dim, dim_div_t<power_dim, area_dim>>);
    }
}

TEST_CASE("Complex dimension combinations") {

    SUBCASE("Pressure * Volume = Energy") { static_assert(std::is_same_v<pressure_volume_dim, energy_dim>); }

    SUBCASE("Force * Time = Momentum") { static_assert(std::is_same_v<force_time_dim, momentum_dim>); }

    SUBCASE("Power / Velocity = Force") { static_assert(std::is_same_v<power_over_velocity_dim, force_dim>); }

    SUBCASE("Energy / (Mass * Temperature) = Specific Heat Capacity") {
        static_assert(std::is_same_v<energy_over_mass_temp_dim, specific_heat_capacity_dim>);
    }
}

TEST_CASE("Dimension arithmetic with fractional exponents") {

    SUBCASE("Square root of area is length") { static_assert(std::is_same_v<dim_root_t<area_dim, 2>, L>); }

    SUBCASE("Cube root of volume is length") { static_assert(std::is_same_v<dim_root_t<volume_dim, 3>, L>); }

    SUBCASE("Square root of energy per mass is velocity") {
        static_assert(std::is_same_v<dim_root_t<energy_per_mass_dim, 2>, velocity_dim>);
    }
}

TEST_CASE("Dimension arithmetic error cases") {

    SUBCASE("Cannot take 0th root") {
        // This should cause a compile-time error
        // Uncomment to test:
        // using invalid_root = root_t<length, 0>;
    }
}
// NOLINTEND