// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/quantity_ops.hpp"
#include "squint/quantity/quantity_types.hpp"
#include "squint/quantity/unit_types.hpp"

using namespace squint;
using namespace squint::units;

TEST_CASE("Unit constructors and value retrieval") {
    SUBCASE("Meter construction and value") {
        meters m(5.0f);
        CHECK(m.value() == doctest::Approx(5.0f));
    }

    SUBCASE("Fahrenheit construction and value") {
        fahrenheit f(32.0f);
        CHECK(f.unit_value() == doctest::Approx(32.0f));
    }
}

TEST_CASE("Unit conversions") {
    SUBCASE("Length conversions") {
        meters m(1.0f);
        CHECK(convert_to<feet_t>(m).unit_value() == doctest::Approx(3.28084f));
        CHECK(convert_to<inches_t>(m).unit_value() == doctest::Approx(39.3701f));
        CHECK(convert_to<kilometers_t>(m).unit_value() == doctest::Approx(0.001f));
        CHECK(convert_to<miles_t>(m).unit_value() == doctest::Approx(0.000621371f));
    }

    SUBCASE("Time conversions") {
        hours h(1.0f);
        CHECK(convert_to<minutes_t>(h).unit_value() == doctest::Approx(60.0f));
        CHECK(convert_to<seconds_t>(h).unit_value() == doctest::Approx(3600.0f));
        CHECK(convert_to<days_t>(h).unit_value() == doctest::Approx(1.0f / 24.0f));
    }

    SUBCASE("Mass conversions") {
        kilograms kg(1.0f);
        CHECK(convert_to<grams_t>(kg).unit_value() == doctest::Approx(1000.0f));
        CHECK(convert_to<pounds_t>(kg).unit_value() == doctest::Approx(2.20462f));
    }

    SUBCASE("Temperature conversions") {
        kelvin k(273.15f);
        CHECK(convert_to<celsius_t>(k).unit_value() == doctest::Approx(0.0f));
        CHECK(convert_to<fahrenheit_t>(k).unit_value() == doctest::Approx(32.0f));
    }

    SUBCASE("Angle conversions") {
        radians r(static_cast<float>(std::numbers::pi));
        CHECK(convert_to<degrees_t>(r).unit_value() == doctest::Approx(180.0f));
    }

    SUBCASE("Velocity conversions") {
        meters_per_second mps(1.0f);
        CHECK(convert_to<kilometers_per_hour_t>(mps).unit_value() == doctest::Approx(3.6f));
        CHECK(convert_to<miles_per_hour_t>(mps).unit_value() == doctest::Approx(2.23694f));
        CHECK(convert_to<feet_per_second_t>(mps).unit_value() == doctest::Approx(3.28084f));
    }

    SUBCASE("Acceleration conversions") {
        meters_per_second_squared mps2(1.0f);
        CHECK(convert_to<feet_per_second_squared_t>(mps2).unit_value() == doctest::Approx(3.28084f));
    }
}

TEST_CASE("Unit arithmetic operations") {
    SUBCASE("Addition") {
        meters m1(2.0f);
        meters m2(3.0f);
        auto result = m1 + m2;
        CHECK(result.value() == doctest::Approx(5.0f));
    }

    SUBCASE("Subtraction") {
        seconds s1(10.0f);
        seconds s2(4.0f);
        auto result = s1 - s2;
        CHECK(result.value() == doctest::Approx(6.0f));
    }

    SUBCASE("Multiplication") {
        meters m(2.0f);
        seconds s(3.0f);
        auto result = m * s;
        CHECK(result.value() == doctest::Approx(6.0f));
    }

    SUBCASE("Division") {
        kilometers km(10.0f);
        CHECK(convert_to<meters_t>(km).unit_value() == doctest::Approx(10000.0f));
        hours h(2.0f);
        auto result = km / h;
        CHECK(convert_to<meters_per_second_t>(result).unit_value() == doctest::Approx(1.38889f));
    }
}

TEST_CASE("Unit arithmetic operations with conversions") {

    SUBCASE("kph, mps, fps") {
        kilometers km(10.0f);
        hours h(2.0f);
        meters_per_second result = km / h;
        CHECK(result.unit_value() == doctest::Approx(1.38889f));
        feet_per_second result2 = result;
        CHECK(result2.unit_value() == doctest::Approx(4.55672f));
    }

    SUBCASE("Implicit conversion from quantity") {
        length l(10.0f);
        feet f(l);
        CHECK(f.unit_value() == doctest::Approx(32.8084f));
        CHECK(f.value() == doctest::Approx(10.0f));
    }

    SUBCASE("Convert to invalid type (compile-time error)") {
        length l(10.0f);
        meters m(l);
        feet f(m);
        CHECK(l.value() == doctest::Approx(10.0f));
        CHECK(m.value() == doctest::Approx(10.0f));
        CHECK(m.unit_value() == doctest::Approx(10.0f));
        CHECK(f.value() == doctest::Approx(10.0f));
        CHECK(f.unit_value() == doctest::Approx(32.8084f));

        // should not compile (uncomment to test)
        // seconds s = convert_to<seconds_t>(l); // convert length to seconds
        // seconds s2 = l; // assign length to seconds
        // seconds s3(l); // construct seconds from length
    }
}

TEST_CASE("Unit comparison operations") {
    SUBCASE("Equality") {
        meters m1(5.0f);
        meters m2(5.0f);
        meters m3(6.0f);
        CHECK(m1 == m2);
        CHECK_FALSE(m1 == m3);
    }

    SUBCASE("Inequality") {
        seconds s1(10.0f);
        seconds s2(15.0f);
        CHECK(s1 != s2);
    }

    SUBCASE("Less than") {
        kilograms kg1(50.0f);
        kilograms kg2(60.0f);
        CHECK(kg1 < kg2);
    }

    SUBCASE("Greater than") {
        celsius c1(25.0f);
        celsius c2(20.0f);
        CHECK(c1 > c2);
    }

    SUBCASE("Less than or equal to") {
        radians r1(1.0f);
        radians r2(1.0f);
        radians r3(2.0f);
        CHECK(r1 <= r2);
        CHECK(r1 <= r3);
    }

    SUBCASE("Greater than or equal to") {
        watts w1(100.0f);
        watts w2(100.0f);
        watts w3(90.0f);
        CHECK(w1 >= w2);
        CHECK(w1 >= w3);
    }
}

TEST_CASE("Type aliases") {
    SUBCASE("Length type aliases") {
        CHECK(std::is_same_v<meters, meters_t<float>>);
        CHECK(std::is_same_v<feet, feet_t<float>>);
        CHECK(std::is_same_v<inches, inches_t<float>>);
        CHECK(std::is_same_v<kilometers, kilometers_t<float>>);
        CHECK(std::is_same_v<miles, miles_t<float>>);
    }

    SUBCASE("Time type aliases") {
        CHECK(std::is_same_v<seconds, seconds_t<float>>);
        CHECK(std::is_same_v<minutes, minutes_t<float>>);
        CHECK(std::is_same_v<hours, hours_t<float>>);
        CHECK(std::is_same_v<days, days_t<float>>);
    }

    SUBCASE("Mass type aliases") {
        CHECK(std::is_same_v<kilograms, kilograms_t<float>>);
        CHECK(std::is_same_v<grams, grams_t<float>>);
        CHECK(std::is_same_v<pounds, pounds_t<float>>);
    }

    SUBCASE("Temperature type aliases") {
        CHECK(std::is_same_v<kelvin, kelvin_t<float>>);
        CHECK(std::is_same_v<celsius, celsius_t<float>>);
        CHECK(std::is_same_v<fahrenheit, fahrenheit_t<float>>);
    }

    SUBCASE("Angle type aliases") {
        CHECK(std::is_same_v<radians, radians_t<float>>);
        CHECK(std::is_same_v<degrees, degrees_t<float>>);
    }

    SUBCASE("Velocity type aliases") {
        CHECK(std::is_same_v<meters_per_second, meters_per_second_t<float>>);
        CHECK(std::is_same_v<kilometers_per_hour, kilometers_per_hour_t<float>>);
        CHECK(std::is_same_v<miles_per_hour, miles_per_hour_t<float>>);
        CHECK(std::is_same_v<feet_per_second, feet_per_second_t<float>>);
    }

    SUBCASE("Acceleration type aliases") {
        CHECK(std::is_same_v<meters_per_second_squared, meters_per_second_squared_t<float>>);
        CHECK(std::is_same_v<feet_per_second_squared, feet_per_second_squared_t<float>>);
    }

    SUBCASE("Other physical quantity type aliases") {
        CHECK(std::is_same_v<newtons, newtons_t<float>>);
        CHECK(std::is_same_v<joules, joules_t<float>>);
        CHECK(std::is_same_v<watts, watts_t<float>>);
        CHECK(std::is_same_v<pascals, pascals_t<float>>);
        CHECK(std::is_same_v<amperes, amperes_t<float>>);
    }
}
// NOLINTEND