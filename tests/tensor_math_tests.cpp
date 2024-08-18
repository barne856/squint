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

TEST_CASE("solve_general()") {
    SUBCASE("Square system with 1D B - column major") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2>> B{5.0, 11.0};
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(6.5).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(-0.5).epsilon(1e-6));
    }

    SUBCASE("Square system with 1D B - row major") {
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> A{{1.0, 3.0, 2.0, 4.0}};
        tensor<double, shape<2>> B{5.0, 11.0};
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(6.5).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(-0.5).epsilon(1e-6));
    }

    SUBCASE("Square system with 2D B - column major") {
        tensor<double, shape<2, 2>> A{{1.0, 3.0, 2.0, 4.0}};
        tensor<double, shape<2, 2>> B{{5.0, 11.0, 10.0, 22.0}};
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(B(1, 0) == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(B(0, 1) == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(B(1, 1) == doctest::Approx(4.0).epsilon(1e-6));
    }

    SUBCASE("Square system with 2D B - row major") {
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> B{{5.0, 10.0, 11.0, 22.0}};
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(B(0, 1) == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(B(1, 0) == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(B(1, 1) == doctest::Approx(4.0).epsilon(1e-6));
    }

    SUBCASE("Overdetermined system with 1D B - column major") {
        tensor<double, shape<3, 2>> A{{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}};
        tensor<double, shape<3>> B{14.0, 32.0, 50.0};
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(4.0).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(5.0).epsilon(1e-6));
    }

    SUBCASE("Overdetermined system with 1D B - row major") {
        tensor<double, shape<3, 2>, strides::row_major<shape<3, 2>>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<double, shape<3>> B{14.0, 32.0, 50.0};
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(4.0).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(5.0).epsilon(1e-6));
    }

    SUBCASE("Overdetermined system with 2D B - column major") {
        tensor<double, shape<3, 2>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<double, shape<3, 2>> B{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(B(1, 0) == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(B(0, 1) == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(B(1, 1) == doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("Overdetermined system with 2D B - row major") {
        tensor<double, shape<3, 2>, strides::row_major<shape<3, 2>>> A{{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}};
        tensor<double, shape<3, 2>, strides::row_major<shape<3, 2>>> B{{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}};
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(B(0, 1) == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(B(1, 0) == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(B(1, 1) == doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("Underdetermined system with 1D B - column major") {
        tensor<double, shape<2, 3>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<double, shape<3>> B{14.0, 32.0, 0.0}; // B must be large enough to hold the result
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(16).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(6).epsilon(1e-6));
        CHECK(B(2) == doctest::Approx(-4).epsilon(1e-6));
    }

    SUBCASE("Underdetermined system with 1D B - row major") {
        tensor<double, shape<2, 3>, strides::row_major<shape<2, 3>>> A{{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}};
        tensor<double, shape<3>> B{14.0, 32.0, 0.0}; // B must be large enough to hold the result
        solve_general(A, B);
        CHECK(B(0) == doctest::Approx(16).epsilon(1e-6));
        CHECK(B(1) == doctest::Approx(6).epsilon(1e-6));
        CHECK(B(2) == doctest::Approx(-4).epsilon(1e-6));
    }

    SUBCASE("Underdetermined system with 2D B - column major") {
        tensor<double, shape<2, 3>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<double, shape<3, 2>> B{{14.0, 28.0, 0.0, 32.0, 64.0, 0.0}}; // B must be large enough to hold the result
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(11.6666).epsilon(1e-3));
        CHECK(B(1, 0) == doctest::Approx(4.6666).epsilon(1e-3));
        CHECK(B(2, 0) == doctest::Approx(-2.3333).epsilon(1e-3));
        CHECK(B(0, 1) == doctest::Approx(26.6666).epsilon(1e-3));
        CHECK(B(1, 1) == doctest::Approx(10.6666).epsilon(1e-3));
        CHECK(B(2, 1) == doctest::Approx(-5.3333).epsilon(1e-3));
    }

    SUBCASE("Underdetermined system with 2D B - row major") {
        tensor<double, shape<2, 3>, strides::row_major<shape<2, 3>>> A{{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}};
        tensor<double, shape<3, 2>, strides::row_major<shape<3, 2>>> B{
            {14.0, 32.0, 28.0, 64.0, 0.0, 0.0}}; // B must be large enough to hold the result
        solve_general(A, B);
        CHECK(B(0, 0) == doctest::Approx(11.6666).epsilon(1e-3));
        CHECK(B(1, 0) == doctest::Approx(4.6666).epsilon(1e-3));
        CHECK(B(2, 0) == doctest::Approx(-2.3333).epsilon(1e-3));
        CHECK(B(0, 1) == doctest::Approx(26.6666).epsilon(1e-3));
        CHECK(B(1, 1) == doctest::Approx(10.6666).epsilon(1e-3));
        CHECK(B(2, 1) == doctest::Approx(-5.3333).epsilon(1e-3));
    }

    SUBCASE("Underdetermined system with 2D A row major B transpose") {
        tensor<double, shape<2, 3>, strides::row_major<shape<2, 3>>> A{{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}};
        tensor<double, shape<2, 3>> B{{14, 32, 28, 64, 0, 0}}; // B must be large enough to hold the result
        auto B_t = B.transpose();
        solve_general(A, B_t);
        CHECK(B_t(0, 0) == doctest::Approx(11.6666).epsilon(1e-3));
        CHECK(B_t(1, 0) == doctest::Approx(4.6666).epsilon(1e-3));
        CHECK(B_t(2, 0) == doctest::Approx(-2.3333).epsilon(1e-3));
        CHECK(B_t(0, 1) == doctest::Approx(26.6666).epsilon(1e-3));
        CHECK(B_t(1, 1) == doctest::Approx(10.6666).epsilon(1e-3));
        CHECK(B_t(2, 1) == doctest::Approx(-5.3333).epsilon(1e-3));
    }
}

// NOLINTEND