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

TEST_CASE("inv()") {
    SUBCASE("2x2 matrix - column major") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto A_inv = inv(A);
        CHECK(A_inv(0, 0) == doctest::Approx(-2.0));
        CHECK(A_inv(0, 1) == doctest::Approx(1.5));
        CHECK(A_inv(1, 0) == doctest::Approx(1.0));
        CHECK(A_inv(1, 1) == doctest::Approx(-0.5));
    }

    SUBCASE("2x2 matrix - row major") {
        tensor<double, shape<2, 2>, strides::row_major<shape<2, 2>>> A{{1.0, 3.0, 2.0, 4.0}};
        auto A_inv = inv(A);
        CHECK(A_inv(0, 0) == doctest::Approx(-2.0));
        CHECK(A_inv(0, 1) == doctest::Approx(1.5));
        CHECK(A_inv(1, 0) == doctest::Approx(1.0));
        CHECK(A_inv(1, 1) == doctest::Approx(-0.5));
    }

    SUBCASE("3x3 matrix - column major") {
        tensor<float, shape<3, 3>> A{{1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f, 6.0f, 0.0f}};
        auto A_inv = inv(A);
        CHECK(A_inv(0, 0) == doctest::Approx(-24.0f));
        CHECK(A_inv(0, 1) == doctest::Approx(20.0f));
        CHECK(A_inv(0, 2) == doctest::Approx(-5.0f));
        CHECK(A_inv(1, 0) == doctest::Approx(18.0f));
        CHECK(A_inv(1, 1) == doctest::Approx(-15.0f));
        CHECK(A_inv(1, 2) == doctest::Approx(4.0f));
        CHECK(A_inv(2, 0) == doctest::Approx(5.0f));
        CHECK(A_inv(2, 1) == doctest::Approx(-4.0f));
        CHECK(A_inv(2, 2) == doctest::Approx(1.0f));
    }

    SUBCASE("3x3 matrix - row major") {
        tensor<float, shape<3, 3>, strides::row_major<shape<3, 3>>> A{
            {1.0f, 0.0f, 5.0f, 2.0f, 1.0f, 6.0f, 3.0f, 4.0f, 0.0f}};
        auto A_inv = inv(A);
        CHECK(A_inv(0, 0) == doctest::Approx(-24.0f));
        CHECK(A_inv(0, 1) == doctest::Approx(20.0f));
        CHECK(A_inv(0, 2) == doctest::Approx(-5.0f));
        CHECK(A_inv(1, 0) == doctest::Approx(18.0f));
        CHECK(A_inv(1, 1) == doctest::Approx(-15.0f));
        CHECK(A_inv(1, 2) == doctest::Approx(4.0f));
        CHECK(A_inv(2, 0) == doctest::Approx(5.0f));
        CHECK(A_inv(2, 1) == doctest::Approx(-4.0f));
        CHECK(A_inv(2, 2) == doctest::Approx(1.0f));
    }

    SUBCASE("Singular matrix - should throw") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 2.0, 4.0}};
        CHECK_THROWS_AS(inv(A), std::runtime_error);
    }

    SUBCASE("Identity matrix") {
        tensor<double, shape<3, 3>> I{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        auto I_inv = inv(I);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(I_inv(i, j) == doctest::Approx(I(i, j)));
            }
        }
    }

    SUBCASE("Inverting a transpose view") {
        tensor<double, shape<2, 2>> A{{1, 3, 2, 4}};
        auto A_T = A.transpose();
        auto A_T_inv = inv(A_T);
        CHECK(A_T_inv(0, 0) == doctest::Approx(-2.0));
        CHECK(A_T_inv(0, 1) == doctest::Approx(1.5));
        CHECK(A_T_inv(1, 0) == doctest::Approx(1.0));
        CHECK(A_T_inv(1, 1) == doctest::Approx(-0.5));
    }
}

TEST_CASE("pinv()") {
    SUBCASE("Square invertible matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto A_pinv = pinv(A);
        auto I = A * A_pinv;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                if (i == j) {
                    CHECK(I(i, j) == doctest::Approx(1.0).epsilon(1e-6));
                } else {
                    CHECK(I(i, j) == doctest::Approx(0.0).epsilon(1e-6));
                }
            }
        }
    }

    SUBCASE("Square non-invertible matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 2.0, 4.0}};
        CHECK_THROWS_AS(pinv(A), std::runtime_error);
    }

    SUBCASE("Overdetermined system") {
        tensor<float, shape<3, 2>> A{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
        auto A_pinv = pinv(A);
        auto AAA = A * A_pinv * A;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                CHECK(AAA(i, j) == doctest::Approx(A(i, j)).epsilon(1e-5));
            }
        }
    }

    SUBCASE("Underdetermined system") {
        tensor<double, shape<2, 3>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        auto A_pinv = pinv(A);
        auto AAA = A * A_pinv * A;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(AAA(i, j) == doctest::Approx(A(i, j)).epsilon(1e-6));
            }
        }
    }

    SUBCASE("Row-major layout") {
        tensor<double, shape<2, 3>, strides::row_major<shape<2, 3>>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        auto A_pinv = pinv(A);
        auto AAA = A * A_pinv * A;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(AAA(i, j) == doctest::Approx(A(i, j)).epsilon(1e-6));
            }
        }
    }

    SUBCASE("1D vector") {
        tensor<double, shape<3>> v{1.0, 2.0, 3.0};
        auto v_pinv = pinv(v);
        auto vvv = v * v_pinv * v;
        for (int i = 0; i < 3; ++i) {
            CHECK(vvv(i) == doctest::Approx(v(i)).epsilon(1e-6));
        }
    }
}

// NOLINTEND