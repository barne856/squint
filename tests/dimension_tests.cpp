#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/dimension.hpp"

using namespace squint;

TEST_CASE("Rational concept") {
    CHECK(rational<std::ratio<1, 2>>);
    CHECK(rational<std::ratio<0>>);
    CHECK_FALSE(rational<int>);
    CHECK_FALSE(rational<double>);
}

TEST_CASE("Dimensional concept") {
    CHECK(dimensional<dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,
                                std::ratio<0>, std::ratio<0>>>);
    CHECK_FALSE(dimensional<int>);
    CHECK_FALSE(dimensional<std::ratio<1>>);
}

TEST_CASE("Dimension struct") {
    using test_dim = dimension<std::ratio<1>, std::ratio<2>, std::ratio<3>, std::ratio<4>, std::ratio<5>, std::ratio<6>,
                               std::ratio<7>>;

    CHECK(std::is_same_v<test_dim::L, std::ratio<1>>);
    CHECK(std::is_same_v<test_dim::T, std::ratio<2>>);
    CHECK(std::is_same_v<test_dim::M, std::ratio<3>>);
    CHECK(std::is_same_v<test_dim::K, std::ratio<4>>);
    CHECK(std::is_same_v<test_dim::I, std::ratio<5>>);
    CHECK(std::is_same_v<test_dim::N, std::ratio<6>>);
    CHECK(std::is_same_v<test_dim::J, std::ratio<7>>);
}

TEST_CASE("Dimension multiplication") {
    using dim1 = dimension<std::ratio<1>, std::ratio<2>, std::ratio<3>, std::ratio<4>, std::ratio<5>, std::ratio<6>,
                           std::ratio<7>>;
    using dim2 = dimension<std::ratio<7>, std::ratio<6>, std::ratio<5>, std::ratio<4>, std::ratio<3>, std::ratio<2>,
                           std::ratio<1>>;
    using result = mult_t<dim1, dim2>;

    CHECK(std::is_same_v<result::L, std::ratio<8>>);
    CHECK(std::is_same_v<result::T, std::ratio<8>>);
    CHECK(std::is_same_v<result::M, std::ratio<8>>);
    CHECK(std::is_same_v<result::K, std::ratio<8>>);
    CHECK(std::is_same_v<result::I, std::ratio<8>>);
    CHECK(std::is_same_v<result::N, std::ratio<8>>);
    CHECK(std::is_same_v<result::J, std::ratio<8>>);
}

TEST_CASE("Dimension division") {
    using dim1 = dimension<std::ratio<8>, std::ratio<8>, std::ratio<8>, std::ratio<8>, std::ratio<8>, std::ratio<8>,
                           std::ratio<8>>;
    using dim2 = dimension<std::ratio<3>, std::ratio<3>, std::ratio<3>, std::ratio<3>, std::ratio<3>, std::ratio<3>,
                           std::ratio<3>>;
    using result = squint::div_t<dim1, dim2>;

    CHECK(std::is_same_v<result::L, std::ratio<5>>);
    CHECK(std::is_same_v<result::T, std::ratio<5>>);
    CHECK(std::is_same_v<result::M, std::ratio<5>>);
    CHECK(std::is_same_v<result::K, std::ratio<5>>);
    CHECK(std::is_same_v<result::I, std::ratio<5>>);
    CHECK(std::is_same_v<result::N, std::ratio<5>>);
    CHECK(std::is_same_v<result::J, std::ratio<5>>);
}

TEST_CASE("Dimension power") {
    using base_dim = dimension<std::ratio<1>, std::ratio<2>, std::ratio<3>, std::ratio<4>, std::ratio<5>, std::ratio<6>,
                               std::ratio<7>>;
    using squared = pow_t<base_dim, 2>;
    using cubed = pow_t<base_dim, 3>;

    CHECK(std::is_same_v<squared::L, std::ratio<2>>);
    CHECK(std::is_same_v<squared::T, std::ratio<4>>);
    CHECK(std::is_same_v<squared::M, std::ratio<6>>);
    CHECK(std::is_same_v<squared::K, std::ratio<8>>);
    CHECK(std::is_same_v<squared::I, std::ratio<10>>);
    CHECK(std::is_same_v<squared::N, std::ratio<12>>);
    CHECK(std::is_same_v<squared::J, std::ratio<14>>);

    CHECK(std::is_same_v<cubed::L, std::ratio<3>>);
    CHECK(std::is_same_v<cubed::T, std::ratio<6>>);
    CHECK(std::is_same_v<cubed::M, std::ratio<9>>);
    CHECK(std::is_same_v<cubed::K, std::ratio<12>>);
    CHECK(std::is_same_v<cubed::I, std::ratio<15>>);
    CHECK(std::is_same_v<cubed::N, std::ratio<18>>);
    CHECK(std::is_same_v<cubed::J, std::ratio<21>>);
}

TEST_CASE("Dimension root") {
    using base_dim = dimension<std::ratio<2>, std::ratio<4>, std::ratio<6>, std::ratio<8>, std::ratio<10>,
                               std::ratio<12>, std::ratio<14>>;
    using sqrt_dim = root_t<base_dim, 2>;

    CHECK(std::is_same_v<sqrt_dim::L, std::ratio<1>>);
    CHECK(std::is_same_v<sqrt_dim::T, std::ratio<2>>);
    CHECK(std::is_same_v<sqrt_dim::M, std::ratio<3>>);
    CHECK(std::is_same_v<sqrt_dim::K, std::ratio<4>>);
    CHECK(std::is_same_v<sqrt_dim::I, std::ratio<5>>);
    CHECK(std::is_same_v<sqrt_dim::N, std::ratio<6>>);
    CHECK(std::is_same_v<sqrt_dim::J, std::ratio<7>>);
}

TEST_CASE("Dimension inverse") {
    using base_dim = dimension<std::ratio<1>, std::ratio<2>, std::ratio<3>, std::ratio<4>, std::ratio<5>, std::ratio<6>,
                               std::ratio<7>>;
    using inv_dim = inv_t<base_dim>;

    CHECK(std::is_same_v<inv_dim::L, std::ratio<-1>>);
    CHECK(std::is_same_v<inv_dim::T, std::ratio<-2>>);
    CHECK(std::is_same_v<inv_dim::M, std::ratio<-3>>);
    CHECK(std::is_same_v<inv_dim::K, std::ratio<-4>>);
    CHECK(std::is_same_v<inv_dim::I, std::ratio<-5>>);
    CHECK(std::is_same_v<inv_dim::N, std::ratio<-6>>);
    CHECK(std::is_same_v<inv_dim::J, std::ratio<-7>>);
}

TEST_CASE("Predefined dimensions") {
    using namespace dimensions;

    CHECK(std::is_same_v<dimensionless, dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,
                                                  std::ratio<0>, std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<length, dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,
                                           std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<dimensions::time, dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>,
                                                     std::ratio<0>, std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<mass, dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>,
                                         std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<temperature, dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>,
                                                std::ratio<0>, std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<current, dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>,
                                            std::ratio<0>, std::ratio<0>>>);
    CHECK(std::is_same_v<amount_of_substance, dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,
                                                        std::ratio<0>, std::ratio<1>, std::ratio<0>>>);
    CHECK(std::is_same_v<luminous_intensity, dimension<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,
                                                       std::ratio<0>, std::ratio<0>, std::ratio<1>>>);
}

TEST_CASE("Derived dimensions") {
    using namespace dimensions;

    CHECK(std::is_same_v<velocity, squint::div_t<length, dimensions::time>>);
    CHECK(std::is_same_v<acceleration, squint::div_t<velocity, dimensions::time>>);
    CHECK(std::is_same_v<force, mult_t<mass, acceleration>>);
    CHECK(std::is_same_v<energy, mult_t<force, length>>);
    CHECK(std::is_same_v<power, squint::div_t<energy, dimensions::time>>);
    CHECK(std::is_same_v<pressure, squint::div_t<force, area>>);
    CHECK(std::is_same_v<frequency, inv_t<dimensions::time>>);
    CHECK(std::is_same_v<charge, mult_t<current, dimensions::time>>);
    CHECK(std::is_same_v<voltage, squint::div_t<energy, charge>>);
    CHECK(std::is_same_v<capacitance, squint::div_t<charge, voltage>>);
    CHECK(std::is_same_v<resistance, squint::div_t<voltage, current>>);
    CHECK(std::is_same_v<magnetic_flux, mult_t<voltage, dimensions::time>>);
    CHECK(std::is_same_v<magnetic_flux_density, squint::div_t<magnetic_flux, area>>);
    CHECK(std::is_same_v<inductance, squint::div_t<magnetic_flux, current>>);
}

TEST_CASE("Complex derived dimensions") {
    using namespace squint::dimensions;

    using force_length = mult_t<force, length>;
    // force = M * L * T^-2
    // length = L
    // force_length = M * L^2 * T^-2

    using time_squared = mult_t<dimensions::time, dimensions::time>;
    // time_squared = T^2

    using force_length_per_time_squared = squint::div_t<force_length, time_squared>;
    // force_length_per_time_squared = (M * L^2 * T^-2) / (T^2) = M * L^2 * T^-4

    using temp_sqrt = root_t<temperature, 2>;
    // temp_sqrt = K^(1/2)

    using complex_dim = mult_t<force_length_per_time_squared, temp_sqrt>;
    // complex_dim = (M * L^2 * T^-4) * K^(1/2) = M * L^2 * T^-4 * K^(1/2)

    using expected = dimension<std::ratio<2>, std::ratio<-4>, std::ratio<1>, std::ratio<1, 2>, std::ratio<0>,
                               std::ratio<0>, std::ratio<0>>;

    CHECK(std::is_same_v<complex_dim, expected>);

    // Let's also check each component individually
    CHECK(std::is_same_v<complex_dim::L, std::ratio<2>>);
    CHECK(std::is_same_v<complex_dim::T, std::ratio<-4>>);
    CHECK(std::is_same_v<complex_dim::M, std::ratio<1>>);
    CHECK(std::is_same_v<complex_dim::K, std::ratio<1, 2>>);
    CHECK(std::is_same_v<complex_dim::I, std::ratio<0>>);
    CHECK(std::is_same_v<complex_dim::N, std::ratio<0>>);
    CHECK(std::is_same_v<complex_dim::J, std::ratio<0>>);
}