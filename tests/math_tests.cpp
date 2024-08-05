// NOLINTBEGIN
#include "squint/quantity/dimension_types.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/math.hpp"
#include "squint/quantity/unit.hpp"
#include "squint/quantity/unit_types.hpp"

using namespace squint;
using namespace squint::units;

TEST_CASE("approx_equal") {
    SUBCASE("quantities") {
        meters m1(5.0);
        meters m2(5.000001);
        meters m3(6.0);
        CHECK(approx_equal(m1, m2));
        CHECK_FALSE(approx_equal(m1, m3));
    }

    SUBCASE("mixed types") {
        squint::units::dimensionless d(1.0);
        CHECK(approx_equal(d, 1.0F));
        CHECK(approx_equal(1.0F, d));
    }
}

TEST_CASE("abs") {
    SUBCASE("quantities") {
        meters m(-5.0);
        CHECK(abs(m).value() == doctest::Approx(5.0));
    }

    SUBCASE("arithmetic") { CHECK(abs(-5.0) == doctest::Approx(5.0)); }
}

TEST_CASE("sqrt") {
    SUBCASE("quantities") {
        square_meters area(25.0);
        auto result = sqrt(area);
        CHECK(result.value() == doctest::Approx(5.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, squint::dimensions::length>);
    }

    SUBCASE("arithmetic") { CHECK(sqrt(25.0) == doctest::Approx(5.0)); }
}

TEST_CASE("root") {
    SUBCASE("quantities") {
        cubic_meters volume(27.0);
        auto result = root<3>(volume);
        CHECK(result.value() == doctest::Approx(3.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, squint::dimensions::length>);
    }

    SUBCASE("arithmetic") { CHECK(root<3>(27.0) == doctest::Approx(3.0)); }
}

TEST_CASE("exp") {
    SUBCASE("dimensionless quantities") {
        squint::units::dimensionless d(1.0);
        CHECK(exp(d).value() == doctest::Approx(std::exp(1.0)));
    }

    SUBCASE("arithmetic") { CHECK(exp(1.0) == doctest::Approx(std::exp(1.0))); }
}

TEST_CASE("log") {
    SUBCASE("dimensionless quantities") {
        squint::units::dimensionless d(std::exp(1.0));
        CHECK(log(d).value() == doctest::Approx(1.0));
    }

    SUBCASE("arithmetic") { CHECK(log(std::exp(1.0)) == doctest::Approx(1.0)); }
}

TEST_CASE("pow") {
    SUBCASE("quantities") {
        meters m(2.0);
        auto result = pow<3>(m);
        CHECK(result.value() == doctest::Approx(8.0));
        CHECK(std::is_same_v<decltype(result)::dimension_type, squint::dimensions::volume>);
    }
}

TEST_CASE("Trigonometric functions") {
    SUBCASE("sin") {
        radians r(M_PI / 2);
        CHECK(sin(r).value() == doctest::Approx(1.0));
        CHECK(sin(M_PI / 2) == doctest::Approx(1.0));
    }

    SUBCASE("cos") {
        radians r(M_PI);
        CHECK(cos(r).value() == doctest::Approx(-1.0));
        CHECK(cos(M_PI) == doctest::Approx(-1.0));
    }

    SUBCASE("tan") {
        radians r(M_PI / 4);
        CHECK(tan(r).value() == doctest::Approx(1.0));
        CHECK(tan(M_PI / 4) == doctest::Approx(1.0));
    }
}

TEST_CASE("Inverse trigonometric functions") {
    SUBCASE("asin") {
        squint::units::dimensionless d(1.0);
        CHECK(asin(d).value() == doctest::Approx(M_PI / 2));
        CHECK(asin(1.0) == doctest::Approx(M_PI / 2));
    }

    SUBCASE("acos") {
        squint::units::dimensionless d(0.0);
        CHECK(acos(d).value() == doctest::Approx(M_PI / 2));
        CHECK(acos(0.0) == doctest::Approx(M_PI / 2));
    }

    SUBCASE("atan") {
        squint::units::dimensionless d(1.0);
        CHECK(atan(d).value() == doctest::Approx(M_PI / 4));
        CHECK(atan(1.0) == doctest::Approx(M_PI / 4));
    }
}

TEST_CASE("atan2") {
    SUBCASE("quantities") {
        meters y(1.0);
        meters x(1.0);
        CHECK(atan2(y, x).value() == doctest::Approx(M_PI / 4));
    }

    SUBCASE("arithmetic") { CHECK(atan2(1.0, 1.0) == doctest::Approx(M_PI / 4)); }

    SUBCASE("mixed") {
        meters y(1.0);
        meters x(1.0);
        CHECK(atan2(y, x).value() == doctest::Approx(M_PI / 4));
        CHECK(atan2(x, y).value() == doctest::Approx(M_PI / 4));
    }
}

TEST_CASE("Hyperbolic functions") {
    SUBCASE("sinh") {
        squint::units::dimensionless d(1.0);
        CHECK(sinh(d).value() == doctest::Approx(std::sinh(1.0)));
        CHECK(sinh(1.0) == doctest::Approx(std::sinh(1.0)));
    }

    SUBCASE("cosh") {
        squint::units::dimensionless d(1.0);
        CHECK(cosh(d).value() == doctest::Approx(std::cosh(1.0)));
        CHECK(cosh(1.0) == doctest::Approx(std::cosh(1.0)));
    }

    SUBCASE("tanh") {
        squint::units::dimensionless d(1.0);
        CHECK(tanh(d).value() == doctest::Approx(std::tanh(1.0)));
        CHECK(tanh(1.0) == doctest::Approx(std::tanh(1.0)));
    }
}

TEST_CASE("Inverse hyperbolic functions") {
    SUBCASE("asinh") {
        squint::units::dimensionless d(1.0);
        CHECK(asinh(d).value() == doctest::Approx(std::asinh(1.0)));
        CHECK(asinh(1.0) == doctest::Approx(std::asinh(1.0)));
    }

    SUBCASE("acosh") {
        squint::units::dimensionless d(2.0);
        CHECK(acosh(d).value() == doctest::Approx(std::acosh(2.0)));
        CHECK(acosh(2.0) == doctest::Approx(std::acosh(2.0)));
    }

    SUBCASE("atanh") {
        squint::units::dimensionless d(0.5);
        CHECK(atanh(d).value() == doctest::Approx(std::atanh(0.5)));
        CHECK(atanh(0.5) == doctest::Approx(std::atanh(0.5)));
    }
}

// NOLINTEND