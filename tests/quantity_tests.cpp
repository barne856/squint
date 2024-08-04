// NOLINTBEGIN
#include "squint/core/error_checking.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/quantity.hpp"
#include "squint/quantity/unit_types.hpp"

TEST_CASE("quantity class comprehensive tests") {
    using dimensionless = squint::quantity<double, squint::dimensions::dimensionless>;
    using length = squint::quantity<double, squint::dimensions::length>;
    using time = squint::quantity<double, squint::dimensions::time>;
    using velocity = squint::quantity<double, squint::dimensions::velocity>;

    SUBCASE("Constructors") {
        // Default constructor
        CHECK_NOTHROW(dimensionless());
        CHECK_NOTHROW(length());

        // Value constructor
        CHECK_NOTHROW(dimensionless(5.0));
        CHECK_NOTHROW(length(10.0));

        // Copy constructor
        length l1(10.0);
        length l2(l1);
        CHECK(l2.value() == 10.0);

        // Move constructor
        length l3(std::move(l1));
        CHECK(l3.value() == 10.0);

        // Implicit conversion for dimensionless quantities
        dimensionless d1 = 5.0;
        double d_val = d1;
        CHECK(d1.value() == 5.0);
        CHECK(d_val == 5.0);

        // Explicit conversion for non-dimensionless quantities
        length l4 = static_cast<length>(10);
        CHECK(l4.value() == 10.0);

        // Conversion from different arithmetic type
        length l5(10.0f); // float to double
        CHECK(l5.value() == 10.0);

        // These should not compile (uncomment to test):
        // length l6 = 10.0;  // Error: implicit conversion
        // velocity v = 5.0;  // Error: implicit conversion
    }

    SUBCASE("Assignment operators") {
        // Copy assignment
        length l1(10.0);
        length l2;
        l2 = l1;
        CHECK(l2.value() == 10.0);

        // Move assignment
        length l3;
        l3 = length(20.0);
        CHECK(l3.value() == 20.0);

        // Assignment from arithmetic type for dimensionless
        dimensionless d;
        d = 5.0;
        CHECK(d.value() == 5.0);

        // These should not compile (uncomment to test):
        // l1 = 10.0;  // Error: assigning scalar to non-dimensionless quantity
        // d = l1;  // Error: assigning non-dimensionless to dimensionless
    }

    SUBCASE("Conversion constructors and operators") {
        // Implicit conversion constructor for dimensionless
        dimensionless d1 = 5; // int to double
        CHECK(d1.value() == 5.0);

        dimensionless d2 = 5.0f; // float to double
        CHECK(d2.value() == 5.0);

        // Explicit conversion constructor for non-dimensionless
        length l1 = length(10); // int to double
        CHECK(l1.value() == 10.0);

        length l2 = length(10.0f); // float to double
        CHECK(l2.value() == 10.0);

        // Explicit conversion to underlying type
        double d_val = static_cast<double>(d1);
        CHECK(d_val == 5.0);

        double l_val = static_cast<double>(l1);
        CHECK(l_val == 10.0);

        // Implicit conversion for dimensionless quantities
        double implicit_d = d1;
        CHECK(implicit_d == 5.0);

        // Explicit conversion to non-underlying type
        int i = static_cast<int>(l1);

        // This should not compile (uncomment to test):
        // double implicit_l = l1;  // Error: implicit conversion of non-dimensionless quantity
    }

    SUBCASE("Conversion between different error checking policies") {
        using length_checked = squint::quantity<double, squint::dimensions::length, squint::error_checking::enabled>;
        using length_unchecked = squint::quantity<double, squint::dimensions::length, squint::error_checking::disabled>;

        length_unchecked lu(10.0);
        length_checked lc(lu);
        CHECK(lc.value() == 10.0);

        length_unchecked lu2(lc);
        CHECK(lu2.value() == 10.0);
    }

    SUBCASE("Accessors") {
        length l(10.0);

        CHECK(l.value() == 10.0);
        CHECK(*l == 10.0);
        CHECK(l.operator->() == &l.value());

        const length cl(20.0);
        CHECK(cl.value() == 20.0);
        CHECK(*cl == 20.0);
        CHECK(cl.operator->() == &cl.value());

        // Mutable access
        l.value() = 30.0;
        CHECK(l.value() == 30.0);

        *l = 40.0;
        CHECK(l.value() == 40.0);

        // These should not compile (uncomment to test):
        // cl.value() = 50.0;  // Error: assignment to const reference
        // *cl = 60.0;  // Error: assignment to const reference
    }

    SUBCASE("Arithmetic operators") {
        length l1(10.0);
        length l2(20.0);
        time t(5.0);

        // Compound assignment
        l1 += length(5.0);
        CHECK(l1.value() == 15.0);

        l2 -= length(5.0);
        CHECK(l2.value() == 15.0);

        // Multiplication and division with scalars
        l1 *= 2.0;
        CHECK(l1.value() == 30.0);

        l2 /= 2.0;
        CHECK(l2.value() == 7.5);

        // Unary minus
        auto l7 = -l1;
        CHECK(l7.value() == -30.0);

        // These should not compile (uncomment to test):
        // l1 += 5.0;  // Error: compound assignment with scalar
        // l1 -= 5.0;  // Error: compound assignment with scalar
        // l1 /= l2;  // Error: compound assignment with a non-dimensionless scalar
        // l1 *= l2;  // Error: compound assignment with a non-dimensionless scalar
    }

    SUBCASE("Comparison operators") {
        length l1(10.0);
        length l2(20.0);

        CHECK(l1 < l2);
        CHECK(l1 <= l2);
        CHECK(l2 > l1);
        CHECK(l2 >= l1);
        CHECK(l1 != l2);
        CHECK(l1 == length(10.0));

        // Three-way comparison
        CHECK((l1 <=> l2) < 0);
        CHECK((l2 <=> l1) > 0);
        CHECK((l1 <=> length(10.0)) == 0);

        // These should not compile (uncomment to test):
        // CHECK(l1 < 10.0);  // Error: comparing quantity with scalar
        // CHECK(20.0 > l2);  // Error: comparing scalar with quantity
    }

    SUBCASE("Increment and decrement operators") {
        length l(10.0);

        // Pre-increment
        auto l1 = ++l;
        CHECK(l.value() == 11.0);
        CHECK(l1.value() == 11.0);

        // Post-increment
        auto l2 = l++;
        CHECK(l.value() == 12.0);
        CHECK(l2.value() == 11.0);

        // Pre-decrement
        auto l3 = --l;
        CHECK(l.value() == 11.0);
        CHECK(l3.value() == 11.0);

        // Post-decrement
        auto l4 = l--;
        CHECK(l.value() == 10.0);
        CHECK(l4.value() == 11.0);
    }

    SUBCASE("Unit conversion") {
        using namespace squint::units;

        auto m = meters(1.0);
        auto f = m.as<feet<>();
        CHECK(f.value() == doctest::Approx(3.28084));

        auto f2 = feet(3.28084);
        auto m2 = f2.as<meters>();
        CHECK(m2.value() == doctest::Approx(1.0));
    }

    SUBCASE("Error checking") {
        using checked_length = squint::quantity<int, squint::dimensions::length, squint::error_checking::enabled>;

        SUBCASE("Addition overflow") {
            checked_length l1(std::numeric_limits<int>::max());
            checked_length l2(1);
            CHECK_THROWS_AS(l1 += l2, std::overflow_error);
        }

        SUBCASE("Subtraction overflow") {
            checked_length l4(std::numeric_limits<int>::min());
            checked_length l5(1);
            CHECK_THROWS_AS(l4 -= l5, std::overflow_error);
        }

        SUBCASE("Multiplication overflow") {
            checked_length l7(1000000);
            CHECK_THROWS_AS(l7 *= 1000000, std::overflow_error);
        }

        SUBCASE("Division by zero") {
            checked_length l9(1);
            CHECK_THROWS_AS(l9 /= 0, std::domain_error);
        }
    }
}
// NOLINTEND