#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity.hpp"
#include <sstream>

using namespace squint;
using namespace squint::dimensions;

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

    SUBCASE("value() accessor") {
        CHECK(q.value() == doctest::Approx(5.0));
    }

    SUBCASE("Arrow operator") {
        CHECK(*q.operator->() == doctest::Approx(5.0));
    }

    SUBCASE("Dereference operator") {
        CHECK(*q == doctest::Approx(5.0));
    }

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
        auto& result = ++q;
        CHECK(q.value() == 6);
        CHECK(&result == &q);
    }

    SUBCASE("Post-increment") {
        auto result = q++;
        CHECK(q.value() == 6);
        CHECK(result.value() == 5);
    }

    SUBCASE("Pre-decrement") {
        auto& result = --q;
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
        CHECK(std::is_same_v<decltype(acceleration)::dimension_type, squint::div_t<length, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Force (mass * length / time^2)") {
        auto force = m * l / (t * t);
        CHECK(std::is_same_v<decltype(force)::dimension_type, squint::div_t<mult_t<mass, length>, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Energy (mass * length^2 / time^2)") {
        auto energy = m * l * l / (t * t);
        CHECK(std::is_same_v<decltype(energy)::dimension_type, squint::div_t<mult_t<mass, mult_t<length, length>>, mult_t<dimensions::time, dimensions::time>>>);
    }

    SUBCASE("Power (mass * length^2 / time^3)") {
        auto power = m * l * l / (t * t * t);
        CHECK(std::is_same_v<decltype(power)::dimension_type, squint::div_t<mult_t<mass, mult_t<length, length>>, mult_t<dimensions::time, mult_t<dimensions::time, dimensions::time>>>>);
    }
}