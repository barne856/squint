// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/tensor.hpp"

using namespace squint;

TEST_CASE("solve()") {
    SUBCASE("Solve with single rhs") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2>> B{5.0, 6.0};
        auto ipiv = solve(A, B);
        CHECK(B(0) == doctest::Approx(-1));
        CHECK(B(1) == doctest::Approx(2));
    }

    SUBCASE("Solve with single rhs row major") {
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> A{{1.0, 3.0, 2.0, 4.0}};
        tensor<double, shape<2>> B{5.0, 6.0};
        auto ipiv = solve(A, B);
        CHECK(B(0) == doctest::Approx(-1));
        CHECK(B(1) == doctest::Approx(2));
    }

    SUBCASE("Solve with 2 rhs") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2, 2>> B{{5.0, 6.0, 7.0, 8.0}};
        auto ipiv = solve(A, B);
        CHECK(B(0, 0) == doctest::Approx(-1));
        CHECK(B(0, 1) == doctest::Approx(-2));
        CHECK(B(1, 0) == doctest::Approx(2));
        CHECK(B(1, 1) == doctest::Approx(3));
    }

    SUBCASE("Solve with 2 rhs row major") {
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> A{{1.0, 3.0, 2.0, 4.0}};
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> B{{5.0, 7.0, 6.0, 8.0}};
        auto ipiv = solve(A, B);
        CHECK(B(0, 0) == doctest::Approx(-1));
        CHECK(B(0, 1) == doctest::Approx(-2));
        CHECK(B(1, 0) == doctest::Approx(2));
        CHECK(B(1, 1) == doctest::Approx(3));
    }
}

// NOLINTEND