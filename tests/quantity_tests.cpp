// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity/dimension_types.hpp"
#include "squint/quantity/quantity_ops.hpp"
#include "squint/quantity/quantity_types.hpp"

using namespace squint;

TEST_CASE("quantity class comprehensive tests") {

    SUBCASE("Constructors") {
        // Default constructor
        CHECK_NOTHROW(pure());
        CHECK_NOTHROW(length());

        // Value constructor
        CHECK_NOTHROW(pure(5.0));
        CHECK_NOTHROW(length(10.0));

        // Copy constructor
        length l1(10.0);
        length l2(l1);
        CHECK(l2.value() == 10.0);

        // Move constructor
        length l3(std::move(l1));
        CHECK(l3.value() == 10.0);

        // Implicit conversion for dimensionless quantities
        pure d1 = 5.0;
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
        pure d;
        d = 5.0;
        CHECK(d.value() == 5.0);

        // These should not compile (uncomment to test):
        // l1 = 10.0;  // Error: assigning scalar to non-dimensionless quantity
        // d = l1;  // Error: assigning non-dimensionless to dimensionless
    }

    SUBCASE("Conversion constructors and operators") {
        // Implicit conversion constructor for dimensionless
        pure d1 = 5; // int to double
        CHECK(d1.value() == 5.0);

        pure d2 = 5.0f; // float to double
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
        using length_checked = squint::quantity<double, squint::dimensions::L, squint::error_checking::enabled>;
        using length_unchecked = squint::quantity<double, squint::dimensions::L, squint::error_checking::disabled>;

        length_unchecked lu(10.0);
        length_checked lc(lu);
        CHECK(lc.value() == 10.0);

        length_unchecked lu2(lc);
        CHECK(lu2.value() == 10.0);
    }

    SUBCASE("Accessors") {
        length l(10.0);
        CHECK(l.value() == 10.0);

        // should not compile (uncomment to test):
        // l = 20.0; // Error: non dimensionless quantity

        const length cl(20.0);
        CHECK(cl.value() == 20.0);

        // should not compile (uncomment to test):
        // cl = length(30.0); // Error: constant quantity
        // cl.value() = 50.0;  // Error: assignment to const reference

        pure d(5.0);
        CHECK(d.value() == 5.0);

        const pure cd(10.0);
        CHECK(cd.value() == 10.0);

        // mutable accessors
        l.value() = 20.0;
        CHECK(l.value() == 20.0);

        d = 10.0;
        CHECK(d.value() == 10.0);
        d.value() = 20.0;
        CHECK(d.value() == 20.0);
    }

    SUBCASE("Arithmetic operators") {
        length l1(10.0);
        length l2(20.0);
        squint::time t(5.0);

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

        #ifndef _MSC_VER
        // Three-way comparison (bug in MSVC prevents this from compiling)
        CHECK((l1 <=> l2) < 0);
        CHECK((l2 <=> l1) > 0);
        CHECK((l1 <=> length(10.0)) == 0);
        #endif

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

    SUBCASE("Error checking") {
        using checked_length = squint::quantity<int, squint::dimensions::L, squint::error_checking::enabled>;

        SUBCASE("Addition overflow") {
            checked_length l1(std::numeric_limits<int>::max());
            checked_length l2(1);
            CHECK_THROWS_AS(l1 += l2, std::overflow_error);
        }

        SUBCASE("Subtraction underflow") {
            checked_length l4(std::numeric_limits<int>::min());
            checked_length l5(1);
            CHECK_THROWS_AS(l4 -= l5, std::underflow_error);
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

TEST_CASE_TEMPLATE("Quantity Addition", T, checked_quantity_t<int, dimensions::L>,
                   unchecked_quantity_t<int, dimensions::L>) {
    SUBCASE("Basic addition") {
        T q1(5);
        T q2(3);
        auto result = q1 + q2;
        CHECK(result.value() == 8);
    }

    SUBCASE("Addition with different types") {
        T q1(5);
        auto q2 = unchecked_quantity_t<double, dimensions::L>(3.5);
        auto result = q1 + q2;
        CHECK(result.value() == doctest::Approx(8.5));
    }

    SUBCASE("Addition overflow") {
        T q1(std::numeric_limits<int>::max());
        T q2(1);
        if constexpr (std::is_same_v<T, checked_quantity_t<int, dimensions::L>>) {
            CHECK_THROWS_AS(q1 + q2, std::overflow_error);
        } else {
            auto result = q1 + q2;
            CHECK(result.value() == std::numeric_limits<int>::min());
        }
    }
}

TEST_CASE_TEMPLATE("Quantity Subtraction", T, checked_quantity_t<int, dimensions::L>,
                   unchecked_quantity_t<int, dimensions::L>) {
    SUBCASE("Basic subtraction") {
        T q1(5);
        T q2(3);
        auto result = q1 - q2;
        CHECK(result.value() == 2);
    }

    SUBCASE("Subtraction with different types") {
        T q1(5);
        auto q2 = unchecked_quantity_t<double, dimensions::L>(3.5);
        auto result = q1 - q2;
        CHECK(result.value() == doctest::Approx(1.5));
    }

    SUBCASE("Subtraction underflow") {
        T q1(std::numeric_limits<int>::min());
        T q2(1);
        if constexpr (std::is_same_v<T, checked_quantity_t<int, dimensions::L>>) {
            CHECK_THROWS_AS(q1 - q2, std::underflow_error);
        } else {
            auto result = q1 - q2;
            CHECK(result.value() == std::numeric_limits<int>::max());
        }
    }
}

TEST_CASE_TEMPLATE("Quantity Multiplication", T, checked_quantity_t<int, dimensions::L>,
                   unchecked_quantity_t<int, dimensions::L>) {
    SUBCASE("Basic multiplication") {
        T q1(5);
        auto q2 = unchecked_quantity_t<int, dimensions::T>(3);
        auto result = q1 * q2;
        CHECK(result.value() == 15);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type,
                              dimension<std::ratio<1, 1>, std::ratio<1, 1>, std::ratio<0, 1>, std::ratio<0, 1>,
                                        std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>>>));
    }

    SUBCASE("Multiplication with different types") {
        T q1(5);
        auto q2 = unchecked_quantity_t<double, dimensions::T>(3.5);
        auto result = q1 * q2;
        CHECK(result.value() == doctest::Approx(17.5));
        CHECK((std::is_same_v<typename decltype(result)::dimension_type,
                              dimension<std::ratio<1, 1>, std::ratio<1, 1>, std::ratio<0, 1>, std::ratio<0, 1>,
                                        std::ratio<0, 1>, std::ratio<0, 1>, std::ratio<0, 1>>>));
    }

    SUBCASE("Multiplication overflow") {
        T q1(std::numeric_limits<int>::max());
        auto q2 = unchecked_quantity_t<int, dimensions::T>(2);
        if constexpr (std::is_same_v<T, checked_quantity_t<int, dimensions::L>>) {
            CHECK_THROWS_AS(q1 * q2, std::overflow_error);
        } else {
            auto result = q1 * q2;
            CHECK(result.value() == -2);
        }
    }
}

TEST_CASE_TEMPLATE("Quantity Division", T, checked_quantity_t<float, dimensions::L>,
                   unchecked_quantity_t<float, dimensions::L>) {
    SUBCASE("Basic division") {
        T q1(15);
        auto q2 = unchecked_quantity_t<float, dimensions::T>(3);
        auto result = q1 / q2;
        CHECK(result.value() == 5);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::velocity_dim>));
    }

    SUBCASE("Division with different types") {
        T q1(15);
        auto q2 = unchecked_quantity_t<double, dimensions::T>(3.5);
        auto result = q1 / q2;
        CHECK(result.value() == doctest::Approx(4.285714));
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::velocity_dim>));
    }

    SUBCASE("Division by zero") {
        T q1(15);
        if constexpr (std::is_same_v<T, checked_quantity_t<float, dimensions::L>>) {
            CHECK_THROWS_AS(q1 / 0, std::domain_error);
        }
    }

    SUBCASE("Division underflow") {
        T q1(1);
        auto q2 = unchecked_quantity_t<float, dimensions::T>(std::numeric_limits<float>::max());
        if constexpr (std::is_same_v<T, checked_quantity_t<float, dimensions::L>>) {
            CHECK_THROWS_AS(q1 / q2, std::underflow_error);
        }
    }
}

TEST_CASE_TEMPLATE("Scalar and Quantity Multiplication", T, checked_quantity_t<int, dimensions::L>,
                   unchecked_quantity_t<int, dimensions::L>) {
    SUBCASE("Scalar * Quantity") {
        int scalar = 2;
        T q(5);
        auto result = scalar * q;
        CHECK(result.value() == 10);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::L>));
    }

    SUBCASE("Quantity * Scalar") {
        T q(5);
        int scalar = 2;
        auto result = q * scalar;
        CHECK(result.value() == 10);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::L>));
    }

    SUBCASE("Scalar * Quantity overflow") {
        int scalar = 2;
        T q(std::numeric_limits<int>::max());
        if constexpr (std::is_same_v<T, checked_quantity_t<int, dimensions::L>>) {
            CHECK_THROWS_AS(scalar * q, std::overflow_error);
        } else {
            auto result = scalar * q;
            CHECK(result.value() == -2);
        }
    }
}

TEST_CASE_TEMPLATE("Scalar and Quantity Division", T, checked_quantity_t<float, dimensions::L>,
                   unchecked_quantity_t<float, dimensions::L>) {
    SUBCASE("Quantity / Scalar") {
        T q(15);
        float scalar = 3;
        auto result = q / scalar;
        CHECK(result.value() == 5);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::L>));
    }

    SUBCASE("Scalar / Quantity") {
        float scalar = 15;
        auto q = unchecked_quantity_t<float, dimensions::T>(3);
        auto result = scalar / q;
        CHECK(result.value() == 5);
        CHECK((std::is_same_v<typename decltype(result)::dimension_type, dimensions::frequency_dim>));
    }

    SUBCASE("Quantity / Scalar division by zero") {
        T q(15);
        float scalar = 0;
        if constexpr (std::is_same_v<T, checked_quantity_t<float, dimensions::L>>) {
            CHECK_THROWS_AS(q / scalar, std::domain_error);
        }
    }

    SUBCASE("Scalar / Quantity division by zero") {
        float scalar = 15;
        auto q = checked_quantity_t<float, dimensions::T>(0);
        CHECK_THROWS_AS(scalar / q, std::domain_error);
    }

    SUBCASE("Quantity / Scalar underflow") {
        T q(1);
        float scalar = std::numeric_limits<float>::max();
        if constexpr (std::is_same_v<T, checked_quantity_t<float, dimensions::L>>) {
            CHECK_THROWS_AS(q / scalar, std::underflow_error);
        }
    }

    SUBCASE("Scalar / Quantity underflow") {
        float scalar = 1;
        auto q = checked_quantity_t<float, dimensions::T>(std::numeric_limits<float>::max());
        if constexpr (std::is_same_v<T, checked_quantity_t<float, dimensions::L>>) {
            CHECK_THROWS_AS(scalar / q, std::underflow_error);
        }
    }
}

TEST_CASE("Stream Operations") {
    SUBCASE("Output stream") {
        checked_quantity_t<int, dimensions::L> q(42);
        std::ostringstream oss;
        oss << q;
        CHECK(oss.str() == "42");
    }

    SUBCASE("Input stream") {
        checked_quantity_t<int, dimensions::L> q(0);
        std::istringstream iss("42");
        iss >> q;
        CHECK(q.value() == 42);
    }

    SUBCASE("Input stream with invalid input") {
        checked_quantity_t<int, dimensions::L> q(0);
        std::istringstream iss("not_a_number");
        iss >> q;
        CHECK(iss.fail());
    }
}
// NOLINTEND