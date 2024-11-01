// NOLINTBEGIN
#include "squint/quantity/quantity_types.hpp"
#include "squint/tensor/tensor.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity.hpp"
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
    
    SUBCASE("1000 random 4x4 matrices") {
        // Use a fixed seed for deterministic random number generation
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dis(-10.0, 10.0);

        // Create test matrices
        using mat4d = tensor<double, shape<4, 4>>;
        
        for (int test = 0; test < 1000; ++test) {
            // Generate a random matrix with deterministic values
            mat4d A;
            bool singular;
            do {
                singular = false;
                // Fill matrix with random values
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        A(i, j) = dis(gen);
                    }
                }
                
                // Check if matrix is singular by computing determinant
                // For 4x4, we can use the direct formula instead of LU decomposition
                double det = 
                    A(0,0) * (
                        A(1,1) * (A(2,2) * A(3,3) - A(2,3) * A(3,2)) -
                        A(1,2) * (A(2,1) * A(3,3) - A(2,3) * A(3,1)) +
                        A(1,3) * (A(2,1) * A(3,2) - A(2,2) * A(3,1))
                    ) -
                    A(0,1) * (
                        A(1,0) * (A(2,2) * A(3,3) - A(2,3) * A(3,2)) -
                        A(1,2) * (A(2,0) * A(3,3) - A(2,3) * A(3,0)) +
                        A(1,3) * (A(2,0) * A(3,2) - A(2,2) * A(3,0))
                    ) +
                    A(0,2) * (
                        A(1,0) * (A(2,1) * A(3,3) - A(2,3) * A(3,1)) -
                        A(1,1) * (A(2,0) * A(3,3) - A(2,3) * A(3,0)) +
                        A(1,3) * (A(2,0) * A(3,1) - A(2,1) * A(3,0))
                    ) -
                    A(0,3) * (
                        A(1,0) * (A(2,1) * A(3,2) - A(2,2) * A(3,1)) -
                        A(1,1) * (A(2,0) * A(3,2) - A(2,2) * A(3,0)) +
                        A(1,2) * (A(2,0) * A(3,1) - A(2,1) * A(3,0))
                    );
                
                singular = std::abs(det) < 1e-10;
            } while (singular);

            // Compute inverse
            auto A_inv = inv(A);

            // Multiply A * A_inv, should get identity matrix
            auto result = A * A_inv;
            auto I = mat4::eye();

            // Check that result is identity matrix
            const double tolerance = 1e-10;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i == j) {
                        // Diagonal elements should be 1
                        CHECK(result(i, j) == doctest::Approx(1.0).epsilon(tolerance));
                    } else {
                        // Off-diagonal elements should be 0
                        CHECK(result(i, j) == doctest::Approx(0.0).epsilon(tolerance));
                    }
                }
            }

            // Also check A_inv * A = I
            auto result2 = A_inv * A;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i == j) {
                        CHECK(result2(i, j) == doctest::Approx(1.0).epsilon(tolerance));
                    } else {
                        CHECK(result2(i, j) == doctest::Approx(0.0).epsilon(tolerance));
                    }
                }
            }
        }
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

TEST_CASE("det()") {
    SUBCASE("2x2 matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = det(A);
        CHECK(result == doctest::Approx(-2.0));
    }

    SUBCASE("3x3 matrix") {
        tensor<double, shape<3, 3>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}};
        auto result = det(A);
        CHECK(result == doctest::Approx(0.0));
    }

    SUBCASE("4x4 matrix zero det") {
        tensor<double, shape<4, 4>> A{
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}};
        auto result = det(A);
        CHECK(result == doctest::Approx(0.0));
    }

    SUBCASE("4x4 matrix non-zero det") {
        tensor<double, shape<4, 4>> A{{1, 5, 4, 1, 5, 5, 5, 2, 6, 2, 6, 3, 7, 1, 2, 4}};
        auto result = det(A);
        CHECK(result == doctest::Approx(-138));
    }
}

TEST_CASE("cross()") {
    SUBCASE("3D vectors") {
        tensor<double, shape<3>> a{1.0, 2.0, 3.0};
        tensor<double, shape<3>> b{4.0, 5.0, 6.0};
        auto c = cross(a, b);
        CHECK(c(0) == doctest::Approx(-3.0));
        CHECK(c(1) == doctest::Approx(6.0));
        CHECK(c(2) == doctest::Approx(-3.0));
    }

    SUBCASE("3D vectors with output") {
        tensor<double, shape<3>> a{1.0, 2.0, 3.0};
        tensor<double, shape<3>> b{4.0, 5.0, 6.0};
        tensor<double, shape<3>> c;
        cross(a, b, c);
        CHECK(c(0) == doctest::Approx(-3.0));
        CHECK(c(1) == doctest::Approx(6.0));
        CHECK(c(2) == doctest::Approx(-3.0));
    }

    SUBCASE("Invalid dimensions") {
        tensor<length, shape<3>> a{length(1.0), length(2.0)};
        tensor<length, shape<3>> b{length(3.0), length(4.0)};
        tensor<length, shape<3>> result;
        // cross(a, b, result); // should not compile
    }
}

TEST_CASE("dot()") {
    SUBCASE("1D vectors") {
        tensor<double, shape<3>> a{1.0, 2.0, 3.0};
        tensor<double, shape<3>> b{4.0, 5.0, 6.0};
        auto result = dot(a, b);
        CHECK(result == doctest::Approx(32.0));
    }

    SUBCASE("Invalid dimensions") {
        tensor<double, shape<2>> a{1.0, 2.0};
        tensor<double, shape<3>> b{3.0, 4.0, 5.0};
    }
}

TEST_CASE("trace()") {
    SUBCASE("2x2 matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = trace(A);
        CHECK(result == doctest::Approx(5.0));
    }

    SUBCASE("3x3 matrix") {
        tensor<double, shape<3, 3>> A{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}};
        auto result = trace(A);
        CHECK(result == doctest::Approx(15.0));
    }
}

TEST_CASE("norm()") {
    SUBCASE("1D vector") {
        tensor<length, shape<3>> a{length(3.0), length(4.0), length(0.0)};
        auto result = norm(a);
        CHECK(result.value() == doctest::Approx(5.0));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = norm(A);
        CHECK(result == doctest::Approx(5.477225575));
    }
}

TEST_CASE("squared_norm()") {
    SUBCASE("1D vector") {
        tensor<length, shape<3>> a{length(1.0), length(2.0), length(3.0)};
        auto result = squared_norm(a);
        CHECK(result.value() == doctest::Approx(14.0));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = squared_norm(A);
        CHECK(result == doctest::Approx(30.0));
    }
}

TEST_CASE("normalize()") {
    SUBCASE("1D vector") {
        tensor<double, shape<3>> a{1.0, 2.0, 3.0};
        auto result = normalize(a);
        CHECK(result(0) == doctest::Approx(0.2672612419));
        CHECK(result(1) == doctest::Approx(0.5345224838));
        CHECK(result(2) == doctest::Approx(0.8017837257));
    }
    SUBCASE("1D vector length") {
        tensor<length, shape<3>> a{length(1.0), length(2.0), length(3.0)};
        auto result = normalize(a);
        static_assert(dimensionless_scalar<typename decltype(result)::value_type>);
        CHECK(result(0) == doctest::Approx(0.2672612419));
        CHECK(result(1) == doctest::Approx(0.5345224838));
        CHECK(result(2) == doctest::Approx(0.8017837257));
    }
}

TEST_CASE("mean()") {
    SUBCASE("1D vector") {
        tensor<double, shape<4>> a{1.0, 2.0, 3.0, 4.0};
        auto result = mean(a);
        CHECK(result == doctest::Approx(2.5));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = mean(A);
        CHECK(result == doctest::Approx(2.5));
    }
}

TEST_CASE("sum()") {
    SUBCASE("1D vector") {
        tensor<double, shape<3>> a{1.0, 2.0, 3.0};
        auto result = sum(a);
        CHECK(result == doctest::Approx(6.0));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        auto result = sum(A);
        CHECK(result == doctest::Approx(10.0));
    }
}

TEST_CASE("min()") {
    SUBCASE("1D vector") {
        tensor<double, shape<4>> a{4.0, 2.0, 1.0, 3.0};
        auto result = min(a);
        CHECK(result == doctest::Approx(1.0));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{4.0, 2.0, 1.0, 3.0}};
        auto result = min(A);
        CHECK(result == doctest::Approx(1.0));
    }
}

TEST_CASE("max()") {
    SUBCASE("1D vector") {
        tensor<double, shape<4>> a{4.0, 2.0, 1.0, 3.0};
        auto result = max(a);
        CHECK(result == doctest::Approx(4.0));
    }

    SUBCASE("2D matrix") {
        tensor<double, shape<2, 2>> A{{4.0, 2.0, 1.0, 3.0}};
        auto result = max(A);
        CHECK(result == doctest::Approx(4.0));
    }
}

TEST_CASE("approx_equal()") {
    SUBCASE("Equal tensors") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2, 2>> B{{1.0, 2.0, 3.0, 4.0}};
        CHECK(approx_equal(A, B));
    }

    SUBCASE("Almost equal tensors") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2, 2>> B{{1.0000001, 2.0, 3.0, 3.9999999}};
        CHECK(approx_equal(A, B, 1e-6));
    }

    SUBCASE("Different tensors") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<2, 2>> B{{1.0, 2.0, 3.0, 5.0}};
        CHECK_FALSE(approx_equal(A, B));
    }

    SUBCASE("Different shapes") {
        tensor<double, shape<2, 2>> A{{1.0, 2.0, 3.0, 4.0}};
        tensor<double, shape<3>> B{1.0, 2.0, 3.0};
        // should not compile
        // approx_equal(A, B);
    }
}

TEST_CASE("contract()") {
    SUBCASE("Fixed 3D") {
        auto A = tensor<float, shape<2, 2, 2>>::arange(1, 1);
        auto B = tensor<float, shape<2, 2, 2>>::arange(1, 1);
        auto result = contract(A, B, std::index_sequence<1>{}, std::index_sequence<0>{});
        CHECK(result(0, 0, 0, 0) == doctest::Approx(7));
        CHECK(result(1, 0, 0, 0) == doctest::Approx(10));
        CHECK(result(0, 1, 0, 0) == doctest::Approx(19));
        CHECK(result(1, 1, 0, 0) == doctest::Approx(22));
        CHECK(result(0, 0, 1, 0) == doctest::Approx(15));
        CHECK(result(1, 0, 1, 0) == doctest::Approx(22));
        CHECK(result(0, 1, 1, 0) == doctest::Approx(43));
        CHECK(result(1, 1, 1, 0) == doctest::Approx(50));
        CHECK(result(0, 0, 0, 1) == doctest::Approx(23));
        CHECK(result(1, 0, 0, 1) == doctest::Approx(34));
        CHECK(result(0, 1, 0, 1) == doctest::Approx(67));
        CHECK(result(1, 1, 0, 1) == doctest::Approx(78));
        CHECK(result(0, 0, 1, 1) == doctest::Approx(31));
        CHECK(result(1, 0, 1, 1) == doctest::Approx(46));
        CHECK(result(0, 1, 1, 1) == doctest::Approx(91));
        CHECK(result(1, 1, 1, 1) == doctest::Approx(106));
    }
    SUBCASE("Fixed 2D") {
        auto A = tensor<float, shape<2, 2>>::arange(1, 1);
        auto B = tensor<float, shape<2, 2>>::arange(1, 1);
        auto result = contract(A, B, std::index_sequence<1>{}, std::index_sequence<0>{});
        // This should be the same as the matrix multiplication
        auto is_equal_mat = result == A * B;
        bool is_equal = std::all_of(is_equal_mat.begin(), is_equal_mat.end(), [](bool b) { return b; });
        CHECK(is_equal);
        CHECK(result(0, 0) == doctest::Approx(7));
        CHECK(result(1, 0) == doctest::Approx(10));
        CHECK(result(0, 1) == doctest::Approx(15));
        CHECK(result(1, 1) == doctest::Approx(22));
    }
    SUBCASE("Dynamic 3D") {
        auto A = tens::arange(1, 1, {2, 2, 2});
        auto B = tens::arange(1, 1, {2, 2, 2});
        std::vector<std::pair<size_t, size_t>> contraction_pairs = {{1, 0}};
        auto result = contract(A, B, contraction_pairs);
        CHECK(result(0, 0, 0, 0) == doctest::Approx(7));
        CHECK(result(1, 0, 0, 0) == doctest::Approx(10));
        CHECK(result(0, 1, 0, 0) == doctest::Approx(19));
        CHECK(result(1, 1, 0, 0) == doctest::Approx(22));
        CHECK(result(0, 0, 1, 0) == doctest::Approx(15));
        CHECK(result(1, 0, 1, 0) == doctest::Approx(22));
        CHECK(result(0, 1, 1, 0) == doctest::Approx(43));
        CHECK(result(1, 1, 1, 0) == doctest::Approx(50));
        CHECK(result(0, 0, 0, 1) == doctest::Approx(23));
        CHECK(result(1, 0, 0, 1) == doctest::Approx(34));
        CHECK(result(0, 1, 0, 1) == doctest::Approx(67));
        CHECK(result(1, 1, 0, 1) == doctest::Approx(78));
        CHECK(result(0, 0, 1, 1) == doctest::Approx(31));
        CHECK(result(1, 0, 1, 1) == doctest::Approx(46));
        CHECK(result(0, 1, 1, 1) == doctest::Approx(91));
        CHECK(result(1, 1, 1, 1) == doctest::Approx(106));
    }
    SUBCASE("Dynamic 2D") {
        auto A = tens::arange(1, 1, {2, 2});
        auto B = tens::arange(1, 1, {2, 2});
        std::vector<std::pair<size_t, size_t>> contraction_pairs = {{1, 0}};
        auto result = contract(A, B, contraction_pairs);
        // This should be the same as the matrix multiplication
        auto is_equal_mat = result == A * B;
        bool is_equal = std::all_of(is_equal_mat.begin(), is_equal_mat.end(), [](bool b) { return b; });
        CHECK(is_equal);
        CHECK(result(0, 0) == doctest::Approx(7));
        CHECK(result(1, 0) == doctest::Approx(10));
        CHECK(result(0, 1) == doctest::Approx(15));
        CHECK(result(1, 1) == doctest::Approx(22));
    }
}

TEST_CASE("tensor_einsum") {
    SUBCASE("Matrix multiplication (dynamic shape)") {
        auto A = tens::arange(1, 1, {2, 3});
        auto B = tens::arange(1, 1, {3, 2});
        auto result = einsum("ij,jk->ik", A, B);
        CHECK(result.shape() == std::vector<size_t>{2, 2});
        auto is_equal_mat = result == A * B;
        bool is_equal = std::all_of(is_equal_mat.begin(), is_equal_mat.end(), [](bool b) { return b; });
        CHECK(is_equal);
        CHECK(result(0, 0) == doctest::Approx(22));
        CHECK(result(0, 1) == doctest::Approx(49));
        CHECK(result(1, 0) == doctest::Approx(28));
        CHECK(result(1, 1) == doctest::Approx(64));
    }

    SUBCASE("Dot product (dynamic shape)") {
        auto A = tens::arange(1, 1, {3});
        auto B = tens::arange(1, 1, {3});
        auto result = einsum("i,i->", A, B);
        CHECK(dot(A, B) == result(0));
        CHECK(result.shape() == std::vector<size_t>{1});
        CHECK(result(0) == doctest::Approx(14));
    }

    SUBCASE("Outer product (dynamic shape)") {
        auto A = tens::arange(1, 1, {3});
        auto B = tens::arange(1, 1, {2});
        auto result = einsum("i,j->ij", A, B);
        CHECK(result.shape() == std::vector<size_t>{3, 2});
        CHECK(result(0, 0) == doctest::Approx(1));
        CHECK(result(0, 1) == doctest::Approx(2));
        CHECK(result(1, 0) == doctest::Approx(2));
        CHECK(result(1, 1) == doctest::Approx(4));
        CHECK(result(2, 0) == doctest::Approx(3));
        CHECK(result(2, 1) == doctest::Approx(6));
    }

    SUBCASE("Trace (dynamic shape)") {
        auto A = tens::arange(1, 1, {3, 3});
        auto result = einsum("ii->", A);
        CHECK(result.shape() == std::vector<size_t>{1});
        CHECK(result(0) == doctest::Approx(15));
    }

    SUBCASE("Diagonal (dynamic shape)") {
        auto A = tens::arange(1, 1, {3, 3});
        auto result = einsum("ii->i", A);
        CHECK(result.shape() == std::vector<size_t>{3});
        CHECK(result(0) == doctest::Approx(1));
        CHECK(result(1) == doctest::Approx(5));
        CHECK(result(2) == doctest::Approx(9));
    }

    SUBCASE("Permutation (dynamic shape)") {
        auto A = tens::arange(1, 1, {2, 3, 4});
        auto result = einsum("ijk->kji", A);
        CHECK(result.shape() == std::vector<size_t>{4, 3, 2});
        CHECK(result(0, 0, 0) == doctest::Approx(1));
        CHECK(result(3, 2, 1) == doctest::Approx(24));
    }

    SUBCASE("Matrix multiplication (fixed shape)") {
        auto A = ndarr<2, 3>::arange(1, 1);
        auto B = ndarr<3, 2>::arange(1, 1);
        auto result = einsum<seq<I, J>, seq<J, K>, seq<I, K>>(A, B);
        CHECK(result.shape() == std::array<size_t, 2>{2, 2});
        auto is_equal_mat = result == A * B;
        bool is_equal = std::all_of(is_equal_mat.begin(), is_equal_mat.end(), [](bool b) { return b; });
        CHECK(is_equal);
        CHECK(result(0, 0) == doctest::Approx(22));
        CHECK(result(0, 1) == doctest::Approx(49));
        CHECK(result(1, 0) == doctest::Approx(28));
        CHECK(result(1, 1) == doctest::Approx(64));
    }

    SUBCASE("Dot product (fixed shape)") {
        auto A = ndarr<3>::arange(1, 1);
        auto B = ndarr<3>::arange(1, 1);
        auto result = einsum<seq<I>, seq<I>, seq<>>(A, B);
        CHECK(dot(A, B) == result());
        CHECK(result.shape().size() == 1);
        CHECK(result.shape()[0] == 1);
        CHECK(result() == doctest::Approx(14));
    }

    SUBCASE("Outer product (fixed shape)") {
        auto A = ndarr<3>::arange(1, 1);
        auto B = ndarr<2>::arange(1, 1);
        auto result = einsum<seq<I>, seq<J>, seq<I, J>>(A, B);
        CHECK(result.shape() == std::array<size_t, 2>{3, 2});
        CHECK(result(0, 0) == doctest::Approx(1));
        CHECK(result(0, 1) == doctest::Approx(2));
        CHECK(result(1, 0) == doctest::Approx(2));
        CHECK(result(1, 1) == doctest::Approx(4));
        CHECK(result(2, 0) == doctest::Approx(3));
        CHECK(result(2, 1) == doctest::Approx(6));
    }

    SUBCASE("Trace (fixed shape)") {
        auto A = ndarr<3, 3>::arange(1, 1);
        auto result = einsum<seq<I, I>, seq<>>(A);
        CHECK(result.shape() == std::array<size_t, 1>{1});
        CHECK(result(0) == doctest::Approx(15));
    }

    SUBCASE("Diagonal (fixed shape)") {
        auto A = ndarr<3, 3>::arange(1, 1);
        auto result = einsum<seq<I, I>, seq<I>>(A);
        CHECK(result.shape() == std::array<size_t, 1>{3});
        CHECK(result(0) == doctest::Approx(1));
        CHECK(result(1) == doctest::Approx(5));
        CHECK(result(2) == doctest::Approx(9));
    }

    SUBCASE("Permutation (fixed shape)") {
        auto A = ndarr<2, 3, 4>::arange(1, 1);
        auto result = einsum<seq<I, J, K>, seq<K, J, I>>(A);
        CHECK(result.shape() == std::array<size_t, 3>{4, 3, 2});
        CHECK(result(0, 0, 0) == doctest::Approx(1));
        CHECK(result(3, 2, 1) == doctest::Approx(24));
    }
}

// NOLINTEND