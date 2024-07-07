#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity.hpp"
#include <sstream>

using namespace squint;
using namespace squint::dimensions;
using namespace squint::constants;

TEST_CASE("Quantity Construction and Basic Operations") {
    SUBCASE("Default constructor") {
        quantity<double, length> q;
        CHECK(q.value() == doctest::Approx(0.0));
    }

    SUBCASE("Value constructor") {
        quantity<double, length> q(5.0);
        CHECK(q.value() == doctest::Approx(5.0));
    }

    SUBCASE("Copy constructor") {
        quantity<double, length> q1(5.0);
        quantity<double, length> q2(q1);
        CHECK(q2.value() == doctest::Approx(5.0));
    }

    SUBCASE("Move constructor") {
        quantity<double, length> q1(5.0);
        quantity<double, length> q2(std::move(q1));
        CHECK(q2.value() == doctest::Approx(5.0));
    }

    SUBCASE("Assignment operator") {
        quantity<double, length> q1(5.0);
        quantity<double, length> q2;
        q2 = q1;
        CHECK(q2.value() == doctest::Approx(5.0));
    }

    SUBCASE("Move assignment operator") {
        quantity<double, length> q1(5.0);
        quantity<double, length> q2;
        q2 = std::move(q1);
        CHECK(q2.value() == doctest::Approx(5.0));
    }

    SUBCASE("Conversion constructor for arithmetic types") {
        quantity<double, dimensionless> q = 5.0;
        CHECK(q.value() == doctest::Approx(5.0));
    }
}

TEST_CASE("Quantity Accessors and Conversions") {
    quantity<double, length> q(5.0);

    SUBCASE("value() accessor") { CHECK(q.value() == doctest::Approx(5.0)); }

    SUBCASE("Arrow operator") { CHECK(*q.operator->() == doctest::Approx(5.0)); }

    SUBCASE("Dereference operator") { CHECK(*q == doctest::Approx(5.0)); }

    SUBCASE("Explicit conversion to value type") {
        double d = static_cast<double>(q);
        CHECK(d == doctest::Approx(5.0));
    }
}

TEST_CASE("Quantity Arithmetic Operations") {
    quantity<double, length> l1(5.0);
    quantity<double, length> l2(3.0);
    quantity<double, dimensions::time> t(2.0);

    SUBCASE("Addition") {
        auto result = l1 + l2;
        CHECK(result.value() == doctest::Approx(8.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Subtraction") {
        auto result = l1 - l2;
        CHECK(result.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Multiplication") {
        auto result = l1 * t;
        CHECK(result.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, mult_t<length, dimensions::time>>);
    }

    SUBCASE("Division") {
        auto result = l1 / t;
        CHECK(result.value() == doctest::Approx(2.5));
        CHECK(std::is_same_v<decltype(result)::dimension_type, squint::div_t<length, dimensions::time>>);
    }

    SUBCASE("Compound addition") {
        l1 += l2;
        CHECK(l1.value() == doctest::Approx(8.0));
    }

    SUBCASE("Compound subtraction") {
        l1 -= l2;
        CHECK(l1.value() == doctest::Approx(2.0));
    }

    SUBCASE("Compound multiplication") {
        l1 *= 2.0;
        CHECK(l1.value() == doctest::Approx(10.0));
    }

    SUBCASE("Compound division") {
        l1 /= 2.0;
        CHECK(l1.value() == doctest::Approx(2.5));
    }

    SUBCASE("Unary negation") {
        auto result = -l1;
        CHECK(result.value() == doctest::Approx(-5.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }
}

TEST_CASE("Quantity Increment and Decrement Operators") {
    quantity<int, length> q(5);

    SUBCASE("Pre-increment") {
        auto &result = ++q;
        CHECK(q.value() == 6);
        CHECK(&result == &q);
    }

    SUBCASE("Post-increment") {
        auto result = q++;
        CHECK(q.value() == 6);
        CHECK(result.value() == 5);
    }

    SUBCASE("Pre-decrement") {
        auto &result = --q;
        CHECK(q.value() == 4);
        CHECK(&result == &q);
    }

    SUBCASE("Post-decrement") {
        auto result = q--;
        CHECK(q.value() == 4);
        CHECK(result.value() == 5);
    }
}

TEST_CASE("Quantity Comparison Operations") {
    quantity<double, length> l1(5.0);
    quantity<double, length> l2(3.0);
    quantity<double, length> l3(5.0);

    SUBCASE("Three-way comparison") {
        CHECK((l1 <=> l2) > 0);
        CHECK((l2 <=> l1) < 0);
        CHECK((l1 <=> l3) == 0);
    }

    SUBCASE("Equality comparison") {
        CHECK(l1 == l3);
        CHECK(l1 != l2);
    }
}

TEST_CASE("Quantity Scalar Operations") {
    quantity<double, length> l(5.0);

    SUBCASE("Scalar multiplication (quantity * scalar)") {
        auto result = l * 2.0;
        CHECK(result.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Scalar multiplication (scalar * quantity)") {
        auto result = 2.0 * l;
        CHECK(result.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Scalar division (quantity / scalar)") {
        auto result = l / 2.0;
        CHECK(result.value() == doctest::Approx(2.5));
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Scalar division (scalar / quantity)") {
        auto result = 10.0 / l;
        CHECK(result.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, inv_t<length>>);
    }
}

TEST_CASE("Quantity Stream Operators") {
    quantity<double, length> l(5.0);

    SUBCASE("Output stream operator") {
        std::ostringstream oss;
        oss << l;
        CHECK(oss.str() == "5");
    }

    SUBCASE("Input stream operator") {
        std::istringstream iss("10");
        quantity<double, length> q;
        iss >> q;
        CHECK(q.value() == doctest::Approx(10.0));
    }
}

TEST_CASE("Quantity Type Traits and Concepts") {
    SUBCASE("is_quantity type trait") {
        CHECK(is_quantity_v<quantity<double, length>>);
        CHECK_FALSE(is_quantity_v<double>);
    }

    SUBCASE("quantitative concept") {
        CHECK(quantitative<quantity<double, length>>);
        CHECK_FALSE(quantitative<double>);
    }
}

TEST_CASE("Quantity Dimension Correctness") {
    quantity<double, length> l(5.0);
    quantity<double, dimensions::time> t(2.0);
    quantity<double, mass> m(3.0);

    SUBCASE("Velocity (length / time)") {
        auto velocity = l / t;
        CHECK(std::is_same_v<decltype(velocity)::dimension_type, squint::div_t<length, dimensions::time>>);
    }

    SUBCASE("Acceleration (length / time^2)") {
        auto acceleration = l / (t * t);
        CHECK(std::is_same_v<decltype(acceleration)::dimension_type,
                             squint::div_t<length, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Force (mass * length / time^2)") {
        auto force = m * l / (t * t);
        CHECK(std::is_same_v<decltype(force)::dimension_type,
                             squint::div_t<mult_t<mass, length>, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Energy (mass * length^2 / time^2)") {
        auto energy = m * l * l / (t * t);
        CHECK(std::is_same_v<
              decltype(energy)::dimension_type,
              squint::div_t<mult_t<mass, mult_t<length, length>>, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Power (mass * length^2 / time^3)") {
        auto power = m * l * l / (t * t * t);
        CHECK(std::is_same_v<decltype(power)::dimension_type,
                             squint::div_t<mult_t<mass, mult_t<length, length>>,
                                           mult_t<dimensions::time, mult_t<dimensions::time, dimensions::time>>>>);
    }
}

TEST_CASE("Quantity Power and Root Operations") {
    quantity<double, length> l(4.0);

    SUBCASE("Power operation") {
        auto area = l.pow<2>();
        CHECK(area.value() == doctest::Approx(16.0));
        CHECK(std::is_same_v<decltype(area)::dimension_type, dimensions::area>);

        auto volume = l.pow<3>();
        CHECK(volume.value() == doctest::Approx(64.0));
        CHECK(std::is_same_v<decltype(volume)::dimension_type, dimensions::volume>);
    }

    SUBCASE("Root operation") {
        auto root = l.root<2>();
        CHECK(root.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(root)::dimension_type, root_t<length, 2>>);
    }

    SUBCASE("Square root operation") {
        auto sqrt_length = l.sqrt();
        CHECK(sqrt_length.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(sqrt_length)::dimension_type, root_t<length, 2>>);
    }
}

TEST_CASE("Unit Conversion") {
    using namespace squint::units;
    SUBCASE("Length conversion") {
        auto meters = length_t<double>::meters(1.0);
        CHECK(meters.as<feet_t>() == doctest::Approx(3.28084));
        CHECK(meters.as<inches_t>() == doctest::Approx(39.3701));
        CHECK(meters.as<kilometers_t>() == doctest::Approx(0.001));
        CHECK(meters.as<miles_t>() == doctest::Approx(0.000621371));
    }

    SUBCASE("Time conversion") {
        auto seconds = squint::units::time_t<double>::seconds(3600.0);
        CHECK(seconds.as<minutes_t>() == doctest::Approx(60.0));
        CHECK(seconds.as<hours_t>() == doctest::Approx(1.0));
        CHECK(seconds.as<days_t>() == doctest::Approx(1.0 / 24.0));
    }

    SUBCASE("Temperature conversion") {
        auto kelvin = temperature_t<double>::kelvin(273.15);
        CHECK(kelvin.as<celsius_t>() == doctest::Approx(0.0));
        CHECK(kelvin.as<fahrenheit_t>() == doctest::Approx(32.0));
    }
}

TEST_CASE("Physical Constants") {
    SUBCASE("Speed of light") {
        CHECK(constants::si_constants<float>::c.value() == doctest::Approx(299792458.0F));
        CHECK(constants::si_constants<double>::c.value() == doctest::Approx(299792458.0));
        CHECK(std::is_same_v<decltype(constants::si_constants<float>::c)::dimension_type, dimensions::velocity>);
        CHECK(std::is_same_v<decltype(constants::si_constants<double>::c)::dimension_type, dimensions::velocity>);
    }

    SUBCASE("Planck constant") {
        CHECK(constants::si_constants<float>::h.value() == doctest::Approx(6.62607015e-34F));
        CHECK(constants::si_constants<double>::h.value() == doctest::Approx(6.62607015e-34));
        CHECK(std::is_same_v<decltype(constants::si_constants<float>::h)::dimension_type,
                             mult_t<dimensions::energy, dimensions::time>>);
        CHECK(std::is_same_v<decltype(constants::si_constants<double>::h)::dimension_type,
                             mult_t<dimensions::energy, dimensions::time>>);
    }

    SUBCASE("Gravitational constant") {
        CHECK(constants::si_constants<float>::G.value() == doctest::Approx(6.67430e-11F));
        CHECK(constants::si_constants<double>::G.value() == doctest::Approx(6.67430e-11));
        CHECK(std::is_same_v<decltype(constants::si_constants<float>::G)::dimension_type,
                             squint::div_t<mult_t<dimensions::force, dimensions::area>, pow_t<dimensions::mass, 2>>>);
        CHECK(std::is_same_v<decltype(constants::si_constants<double>::G)::dimension_type,
                             squint::div_t<mult_t<dimensions::force, dimensions::area>, pow_t<dimensions::mass, 2>>>);
    }
}

TEST_CASE("Derived Units") {
    using namespace squint::units;
    SUBCASE("Velocity") {
        auto v = length_t<double>::meters(10.0) / squint::units::time_t<double>::seconds(2.0);
        CHECK(v.value() == doctest::Approx(5.0));
        CHECK(std::is_same_v<decltype(v)::dimension_type, dimensions::velocity>);
    }

    SUBCASE("Acceleration") {
        auto a = velocity_t<double>::meters_per_second(10.0) / squint::units::time_t<double>::seconds(2.0);
        CHECK(a.value() == doctest::Approx(5.0));
        CHECK(std::is_same_v<decltype(a)::dimension_type, dimensions::acceleration>);
    }

    SUBCASE("Force") {
        auto f = mass_t<double>::kilograms(2.0) * acceleration_t<double>::meters_per_second_squared(5.0);
        CHECK(f.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(f)::dimension_type, dimensions::force>);
    }
}

TEST_CASE("Quantity Arithmetic with Mixed Types") {
    quantity<double, length> l_double(5.0);
    quantity<float, length> l_float(3.0F);
    quantity<int, length> l_int(2);

    SUBCASE("Addition with mixed types") {
        auto result = l_double + l_float + l_int;
        CHECK(result.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(result)::value_type, double>);
        CHECK(std::is_same_v<decltype(result)::dimension_type, length>);
    }

    SUBCASE("Multiplication with mixed types") {
        auto result = l_double * l_float * l_int;
        CHECK(result.value() == doctest::Approx(30.0));
        CHECK(std::is_same_v<decltype(result)::value_type, double>);
        CHECK(std::is_same_v<decltype(result)::dimension_type, mult_t<length, mult_t<length, length>>>);
    }
}

TEST_CASE("Quantity Operations with Constants") {
    using namespace squint::units;

    SUBCASE("Multiplication with pi") {
        {
            auto circle_circumference = length_t<float>::meters(2.0F) * constants::math_constants<float>::pi;
            CHECK(circle_circumference.value() == doctest::Approx(2.0F * std::numbers::pi_v<float>));
            CHECK(std::is_same_v<decltype(circle_circumference)::dimension_type, dimensions::length>);
        }
        {
            auto circle_circumference = length_t<double>::meters(2.0) * constants::math_constants<double>::pi;
            CHECK(circle_circumference.value() == doctest::Approx(2.0 * std::numbers::pi));
            CHECK(std::is_same_v<decltype(circle_circumference)::dimension_type, dimensions::length>);
        }
    }

    SUBCASE("Division by speed of light") {
        {
            auto time_dilation = squint::units::time_t<float>::seconds(1.0F) / constants::si_constants<float>::c;
            CHECK(time_dilation.value() == doctest::Approx(1.0F / 299792458.0F));
            CHECK(std::is_same_v<decltype(time_dilation)::dimension_type,
                                 squint::div_t<dimensions::time, dimensions::velocity>>);
        }
        {
            auto time_dilation = squint::units::time_t<double>::seconds(1.0) / constants::si_constants<double>::c;
            CHECK(time_dilation.value() == doctest::Approx(1.0 / 299792458.0));
            CHECK(std::is_same_v<decltype(time_dilation)::dimension_type,
                                 squint::div_t<dimensions::time, dimensions::velocity>>);
        }
    }
}

TEST_CASE("Error Handling and Edge Cases") {
    SUBCASE("Integer Overflow") {
        using checked_int_length = quantity<int, dimensions::length, error_checking::enabled>;

        SUBCASE("Addition") {
            checked_int_length max_length(std::numeric_limits<int>::max());
            CHECK_THROWS_AS(max_length + checked_int_length(1), std::overflow_error);

            checked_int_length min_length(std::numeric_limits<int>::min());
            CHECK_THROWS_AS(min_length + checked_int_length(-1), std::overflow_error);
        }

        SUBCASE("Subtraction") {
            checked_int_length max_length(std::numeric_limits<int>::max());
            CHECK_THROWS_AS(max_length - checked_int_length(-1), std::overflow_error);

            checked_int_length min_length(std::numeric_limits<int>::min());
            CHECK_THROWS_AS(min_length - checked_int_length(1), std::overflow_error);
        }

        SUBCASE("Multiplication") {
            checked_int_length large_length(std::numeric_limits<int>::max() / 2 + 1);
            CHECK_THROWS_AS(large_length * 2, std::overflow_error);
            CHECK_THROWS_AS(2 * large_length, std::overflow_error);

            checked_int_length negative_length(std::numeric_limits<int>::min() / 2 - 1);
            CHECK_THROWS_AS(negative_length * 2, std::overflow_error);
            CHECK_THROWS_AS(2 * negative_length, std::overflow_error);
        }
    }

    SUBCASE("Division by Zero") {
        using checked_double_length = quantity<double, dimensions::length, error_checking::enabled>;
        checked_double_length length(10.0);

        CHECK_THROWS_AS(length / 0.0, std::domain_error);
        CHECK_THROWS_AS(length / checked_double_length(0.0), std::domain_error);
        CHECK_THROWS_AS(1.0 / checked_double_length(0.0), std::domain_error);
    }

    SUBCASE("Floating-Point Underflow") {
        using checked_float_length = quantity<float, dimensions::length, error_checking::enabled>;
        checked_float_length tiny_length(std::numeric_limits<float>::min());

        CHECK_THROWS_AS(tiny_length / std::numeric_limits<float>::max(), std::underflow_error);
    }

    SUBCASE("No Error Checking") {
        using unchecked_int_length = quantity<int, dimensions::length>;

        unchecked_int_length max_length(std::numeric_limits<int>::max());
        CHECK_NOTHROW(max_length + unchecked_int_length(1));

        unchecked_int_length min_length(std::numeric_limits<int>::min());
        CHECK_NOTHROW(min_length - unchecked_int_length(1));
    }

    SUBCASE("Edge Cases") {
        SUBCASE("Integer Division Rounding") {
            using int_length = quantity<int, dimensions::length>;
            CHECK((int_length(5) / int_length(2)).value() == 2);
            CHECK((int_length(-5) / int_length(2)).value() == -2);
        }

        SUBCASE("Floating-Point Precision") {
            using float_length = quantity<float, dimensions::length>;
            float_length a(1.0F);
            float_length b(1.0e-8F);
            float_length sum = a + b;
            CHECK(sum.value() == doctest::Approx(1.0F));
        }

        SUBCASE("Negative Zero") {
            using double_length = quantity<double, dimensions::length>;
            double_length pos_zero(0.0);
            double_length neg_zero(-0.0);
            CHECK(pos_zero == neg_zero);
        }

        SUBCASE("NaN Handling") {
            using double_length = quantity<double, dimensions::length>;
            double_length nan_length(std::numeric_limits<double>::quiet_NaN());
            CHECK(nan_length != nan_length);
            CHECK_FALSE(nan_length == nan_length);
        }

        SUBCASE("Infinity Handling") {
            using double_length = quantity<double, dimensions::length>;
            double_length inf_length(std::numeric_limits<double>::infinity());
            CHECK(inf_length > double_length(std::numeric_limits<double>::max()));
            CHECK(-inf_length < double_length(std::numeric_limits<double>::lowest()));
        }
    }
}

TEST_CASE("Mixed Error-Checked and Non-Error-Checked Quantity Operations") {
    using checked_double_length = quantity<double, dimensions::length, error_checking::enabled>;
    using unchecked_double_length = quantity<double, dimensions::length, error_checking::disabled>;
    using checked_int_length = quantity<int, dimensions::length, error_checking::enabled>;
    using unchecked_int_length = quantity<int, dimensions::length, error_checking::disabled>;

    SUBCASE("Addition with mixed error checking") {
        checked_double_length checked(5.0);
        unchecked_double_length unchecked(3.0);
        
        auto result = checked + unchecked;
        CHECK(result.value() == doctest::Approx(8.0));
        CHECK(std::is_same_v<decltype(result), checked_double_length>);

        result = unchecked + checked;
        CHECK(result.value() == doctest::Approx(8.0));
        CHECK(std::is_same_v<decltype(result), checked_double_length>);
    }

    SUBCASE("Subtraction with mixed error checking") {
        checked_double_length checked(5.0);
        unchecked_double_length unchecked(3.0);
        
        auto result = checked - unchecked;
        CHECK(result.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(result), checked_double_length>);

        result = unchecked - checked;
        CHECK(result.value() == doctest::Approx(-2.0));
        CHECK(std::is_same_v<decltype(result), checked_double_length>);
    }

    SUBCASE("Multiplication with mixed error checking") {
        checked_double_length checked(5.0);
        unchecked_double_length unchecked(3.0);
        
        auto result = checked * unchecked;
        CHECK(result.value() == doctest::Approx(15.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, mult_t<dimensions::length, dimensions::length>>);
        CHECK(std::is_same_v<decltype(result)::value_type, double>);
        CHECK(std::is_same_v<decltype(result), quantity<double, mult_t<dimensions::length, dimensions::length>, error_checking::enabled>>);

        result = unchecked * checked;
        CHECK(result.value() == doctest::Approx(15.0));
        CHECK(std::is_same_v<decltype(result), quantity<double, mult_t<dimensions::length, dimensions::length>, error_checking::enabled>>);
    }

    SUBCASE("Division with mixed error checking") {
        checked_double_length checked(6.0);
        unchecked_double_length unchecked(3.0);
        
        auto result = checked / unchecked;
        CHECK(result.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, dimensions::dimensionless>);
        CHECK(std::is_same_v<decltype(result)::value_type, double>);
        CHECK(std::is_same_v<decltype(result), quantity<double, dimensions::dimensionless, error_checking::enabled>>);

        result = unchecked / checked;
        CHECK(result.value() == doctest::Approx(0.5));
        CHECK(std::is_same_v<decltype(result), quantity<double, dimensions::dimensionless, error_checking::enabled>>);
    }

    SUBCASE("Scalar operations with mixed error checking") {
        checked_double_length checked(5.0);
        unchecked_double_length unchecked(3.0);

        auto result1 = checked * 2.0;
        CHECK(result1.value() == doctest::Approx(10.0));
        CHECK(std::is_same_v<decltype(result1), checked_double_length>);

        auto result2 = 2.0 * unchecked;
        CHECK(result2.value() == doctest::Approx(6.0));
        CHECK(std::is_same_v<decltype(result2), unchecked_double_length>);

        auto result3 = checked / 2.0;
        CHECK(result3.value() == doctest::Approx(2.5));
        CHECK(std::is_same_v<decltype(result3), checked_double_length>);

        auto result4 = 6.0 / unchecked;
        CHECK(result4.value() == doctest::Approx(2.0));
        CHECK(std::is_same_v<decltype(result4)::dimension_type, inv_t<dimensions::length>>);
        CHECK(std::is_same_v<decltype(result4)::value_type, double>);
        CHECK(std::is_same_v<decltype(result4), quantity<double, inv_t<dimensions::length>, error_checking::disabled>>);
    }

    SUBCASE("Error checking behavior") {
        checked_int_length checked_max(std::numeric_limits<int>::max());
        unchecked_int_length unchecked_one(1);

        CHECK_THROWS_AS(checked_max + unchecked_one, std::overflow_error);
        CHECK_THROWS_AS(unchecked_one + checked_max, std::overflow_error);

        checked_int_length checked_zero(0);
        unchecked_int_length unchecked_zero(0);

        CHECK_THROWS_AS(checked_zero / unchecked_zero, std::domain_error);
        CHECK_THROWS_AS(unchecked_zero / checked_zero, std::domain_error);
    }

    SUBCASE("Mixed type and error checking") {
        checked_double_length checked_double(5.0);
        unchecked_int_length unchecked_int(3);

        auto result = checked_double + unchecked_int;
        CHECK(result.value() == doctest::Approx(8.0));
        CHECK(std::is_same_v<decltype(result), checked_double_length>);

        auto result2 = unchecked_int * checked_double;
        CHECK(result2.value() == doctest::Approx(15.0));
        CHECK(std::is_same_v<decltype(result2), quantity<double, mult_t<dimensions::length, dimensions::length>, error_checking::enabled>>);
    }
}