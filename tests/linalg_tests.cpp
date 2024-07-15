#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <concepts>

using namespace squint;

TEST_CASE("linear_algebra_mixin tests") {
    SUBCASE("operator+= for fixed tensors") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{5, 6, 7, 8}};
        a += b;
        CHECK(a == mat2{{6, 8, 10, 12}});
    }

    SUBCASE("operator+= for dynamic tensors") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {5, 6, 7, 8}};
        a += b;
        CHECK(a == tens{{2, 2}, {6, 8, 10, 12}});
    }

    SUBCASE("operator-= for fixed tensors") {
        mat2 a{{5, 6, 7, 8}};
        mat2 b{{1, 2, 3, 4}};
        a -= b;
        CHECK(a == mat2{{4, 4, 4, 4}});
    }

    SUBCASE("operator-= for dynamic tensors") {
        auto a = tens{{2, 2}, {5, 6, 7, 8}};
        auto b = tens{{2, 2}, {1, 2, 3, 4}};
        a -= b;
        CHECK(a == tens{{2, 2}, {4, 4, 4, 4}});
    }

    SUBCASE("operator*= scalar") {
        mat2 a{{1, 2, 3, 4}};
        a *= 2;
        CHECK(a == mat2{{2, 4, 6, 8}});
    }

    SUBCASE("operator/= scalar") {
        mat2 a{{2, 4, 6, 8}};
        a /= 2;
        CHECK(a == mat2{{1, 2, 3, 4}});
    }

    SUBCASE("norm") {
        vec3 v{{3, 4, 0}};
        CHECK(v.norm() == doctest::Approx(5.0));
    }

    SUBCASE("squared_norm") {
        vec3 v{{3, 4, 0}};
        CHECK(v.squared_norm() == 25);
    }

    SUBCASE("trace") {
        mat3 m{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        CHECK(m.trace() == 15);
    }

    SUBCASE("mean") {
        vec4 v{{1, 2, 3, 4}};
        CHECK(v.mean() == 2.5);
    }

    SUBCASE("sum") {
        vec4 v{{1, 2, 3, 4}};
        CHECK(v.sum() == 10);
    }

    SUBCASE("min") {
        vec4 v{{4, 2, 3, 1}};
        CHECK(v.min() == 1);
    }

    SUBCASE("max") {
        vec4 v{{4, 2, 3, 1}};
        CHECK(v.max() == 4);
    }
}

TEST_CASE("fixed_linear_algebra_mixin tests") {
    SUBCASE("operator/ (solve)") {
        mat2 A{{4, 3, 2, 1}};
        vec2 b{{20, 10}};
        auto x = b / A;
        CHECK(x[0] == doctest::Approx(0));
        CHECK(x[1] == doctest::Approx(10));
    }

    SUBCASE("operator==") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{1, 2, 3, 4}};
        mat2 c{{5, 6, 7, 8}};
        CHECK(a == b);
        CHECK_FALSE(a == c);
    }

    SUBCASE("operator!=") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{1, 2, 3, 4}};
        mat2 c{{5, 6, 7, 8}};
        CHECK_FALSE(a != b);
        CHECK(a != c);
    }

    SUBCASE("transpose") {
        mat2x3 m{{1, 2, 3, 4, 5, 6}};
        auto mt = m.transpose();
        CHECK(mt == mat3x2{{1, 3, 5, 2, 4, 6}});
    }

    SUBCASE("inv") {
        mat2 m{{4, 7, 2, 6}};
        auto m_inv = m.inv();
        auto identity = m * m_inv;
        CHECK(identity[0, 0] == doctest::Approx(1.0));
        CHECK(identity[0, 1] == doctest::Approx(0.0));
        CHECK(identity[1, 0] == doctest::Approx(0.0));
        CHECK(identity[1, 1] == doctest::Approx(1.0));
    }

    SUBCASE("pinv") {
        mat2x3 m{{1, 2, 3, 4, 5, 6}};
        auto m_pinv = m.pinv();
        auto pseudo_identity = m * m_pinv * m;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(pseudo_identity[i, j] == doctest::Approx(m[i, j]));
            }
        }
    }
}

TEST_CASE("dynamic_linear_algebra_mixin tests") {
    SUBCASE("operator/ (solve)") {
        auto A = tens{{2, 2}, {4, 3, 2, 1}};
        auto b = tens{{2}, {20, 10}};
        auto x = b / A;
        CHECK(x[0] == doctest::Approx(0));
        CHECK(x[1] == doctest::Approx(10));
    }

    SUBCASE("operator==") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {1, 2, 3, 4}};
        auto c = tens{{2, 2}, {5, 6, 7, 8}};
        CHECK(a == b);
        CHECK_FALSE(a == c);
    }

    SUBCASE("operator!=") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {1, 2, 3, 4}};
        auto c = tens{{2, 2}, {5, 6, 7, 8}};
        CHECK_FALSE(a != b);
        CHECK(a != c);
    }

    SUBCASE("transpose") {
        auto m = tens{{2, 3}, {1, 2, 3, 4, 5, 6}};
        auto mt = m.transpose();
        CHECK(mt == tens{{3, 2}, {1, 3, 5, 2, 4, 6}});
    }

    SUBCASE("inv") {
        auto m = tens{{2, 2}, {4, 7, 2, 6}};
        auto m_inv = m.inv();
        auto identity = m * m_inv;
        CHECK(identity[0, 0] == doctest::Approx(1.0));
        CHECK(identity[0, 1] == doctest::Approx(0.0));
        CHECK(identity[1, 0] == doctest::Approx(0.0));
        CHECK(identity[1, 1] == doctest::Approx(1.0));
    }

    SUBCASE("pinv") {
        auto m = tens{{2, 3}, {1, 2, 3, 4, 5, 6}};
        auto m_pinv = m.pinv();
        auto pseudo_identity = m * m_pinv * m;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(pseudo_identity[i, j] == doctest::Approx(m[i, j]));
            }
        }
    }
}

TEST_CASE("Element-wise operations") {
    SUBCASE("Fixed tensors") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{5, 6, 7, 8}};
        CHECK(a + b == mat2{{6, 8, 10, 12}});
        CHECK(a - b == mat2{{-4, -4, -4, -4}});
    }

    SUBCASE("Dynamic tensors") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {5, 6, 7, 8}};
        CHECK(a + b == tens{{2, 2}, {6, 8, 10, 12}});
        CHECK(a - b == tens{{2, 2}, {-4, -4, -4, -4}});
    }
}

TEST_CASE("Matrix multiplication") {
    SUBCASE("Fixed tensors") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{5, 6, 7, 8}};
        CHECK(a * b == mat2{{23, 34, 31, 46}});
    }

    SUBCASE("Dynamic tensors") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {5, 6, 7, 8}};
        CHECK(a * b == tens{{2, 2}, {23, 34, 31, 46}});
    }
}

TEST_CASE("Solve linear system") {
    SUBCASE("Fixed tensors") {
        mat2 A{{4, 3, 2, 1}};
        vec2 b{{20, 10}};
        auto ipiv = solve(A, b);
        CHECK(b[0] == doctest::Approx(0));
        CHECK(b[1] == doctest::Approx(10));
    }

    SUBCASE("Dynamic tensors") {
        auto A = tens{{2, 2}, {4, 3, 2, 1}};
        auto b = tens{{2}, {20, 10}};
        auto ipiv = solve(A, b);
        CHECK(b[0] == doctest::Approx(0));
        CHECK(b[1] == doctest::Approx(10));
    }
}

TEST_CASE("Solve linear least squares") {
    SUBCASE("Fixed tensors") {
        mat2x3 A{{1, 2, 3, 4, 5, 6}};
        vec3 b{{7, 8, 0}};
        solve_lls(A, b);
        CHECK(b[0] == doctest::Approx(-0.66666).epsilon(0.01));
        CHECK(b[1] == doctest::Approx(0.33333).epsilon(0.01));
        CHECK(b[2] == doctest::Approx(1.33333).epsilon(0.01));
    }

    SUBCASE("Dynamic tensors") {
        auto A = tens{{2, 3}, {1, 2, 3, 4, 5, 6}};
        auto b = tens{{3}, {7, 8, 0}};
        solve_lls(A, b);
        CHECK(b[0] == doctest::Approx(-0.66666).epsilon(0.01));
        CHECK(b[1] == doctest::Approx(0.33333).epsilon(0.01));
        CHECK(b[2] == doctest::Approx(1.33333).epsilon(0.01));
    }
}

TEST_CASE("Cross product") {
    SUBCASE("Fixed tensors") {
        vec3 a{{1, 2, 3}};
        vec3 b{{4, 5, 6}};
        auto c = cross(a, b);
        CHECK(c == vec3{{-3, 6, -3}});
    }

    SUBCASE("Dynamic tensors") {
        auto a = tens{{3}, {1, 2, 3}};
        auto b = tens{{3}, {4, 5, 6}};
        auto c = cross(a, b);
        CHECK(c == tens{{3}, {-3, 6, -3}});
    }
}

TEST_CASE("Approximate equality") {
    SUBCASE("Fixed tensors") {
        mat2 a{{1, 2, 3, 4}};
        mat2 b{{1.000001, 2.000001, 3.000001, 4.000001}};
        CHECK(approx_equal(a, b, 1e-5F));
    }

    SUBCASE("Dynamic tensors") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        auto b = tens{{2, 2}, {1.000001, 2.000001, 3.000001, 4.000001}};
        CHECK(approx_equal(a, b, 1e-5F));
    }
}

TEST_CASE("Scalar operations") {
    SUBCASE("Fixed tensors") {
        mat2 a{{1, 2, 3, 4}};
        CHECK(a * 2 == mat2{{2, 4, 6, 8}});
        CHECK(2 * a == mat2{{2, 4, 6, 8}});
        CHECK(a / 2 == mat2{{0.5, 1, 1.5, 2}});
    }

    SUBCASE("Dynamic tensors") {
        auto a = tens{{2, 2}, {1, 2, 3, 4}};
        CHECK(a * 2 == tens{{2, 2}, {2, 4, 6, 8}});
        CHECK(2 * a == tens{{2, 2}, {2, 4, 6, 8}});
        CHECK(a / 2 == tens{{2, 2}, {0.5, 1, 1.5, 2}});
    }
}

TEST_CASE("BLAS operations with transposed tensors and views") {
    SUBCASE("Matrix multiplication with transposed fixed tensors") {
        mat2x3 a{{1, 2, 3, 4, 5, 6}};
        mat2x3 b{{7, 8, 9, 10, 11, 12}};
        auto c = a * b.transpose();
        CHECK(c == mat2{{89, 116, 98, 128}});
    }

    SUBCASE("Matrix multiplication with transposed dynamic tensors") {
        auto a = tens{{2, 3}, {1, 2, 3, 4, 5, 6}};
        auto b = tens{{2, 3}, {7, 8, 9, 10, 11, 12}};
        auto c = a * b.transpose();
        CHECK(c == tens{{2, 2}, {89, 116, 98, 128}});
    }

    SUBCASE("Matrix multiplication with fixed tensor views") {
        mat3 a{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        mat3 b{{10, 11, 12, 13, 14, 15, 16, 17, 18}};
        auto a_view = a.subview<2, 2>(slice{0, 2}, slice{0, 2});
        auto b_view = b.subview<2, 2>(slice{1, 2}, slice{1, 2});
        auto c = a_view * b_view;
        CHECK(c == mat2{{74, 103, 89, 124}});
    }

    SUBCASE("Matrix multiplication with dynamic tensor views") {
        auto a = tens{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
        auto b = tens{{3, 3}, {10, 11, 12, 13, 14, 15, 16, 17, 18}};
        auto a_view = a.subview(slice{0, 2}, slice{0, 2});
        auto b_view = b.subview(slice{1, 2}, slice{1, 2});
        auto c = a_view * b_view;
        CHECK(c == mat2{{74, 103, 89, 124}});
    }
}

TEST_CASE("LAPACK operations with transposed tensors and views") {
    SUBCASE("Solve linear least squares with fixed tensor views") {
        mat3 A{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        vec3 b{{14, 32, 50}};
        auto A_view = A.subview<2, 3>(slice{0, 2}, slice{0, 3});
        auto b_view = b.subview<3>(slice{0, 3});
        solve_lls(A_view, b_view);
        CHECK(b[0] == doctest::Approx(15.6666).epsilon(0.01));
        CHECK(b[1] == doctest::Approx(6).epsilon(0.01));
    }

    SUBCASE("Solve linear least squares with dynamic tensor views") {
        auto A = tens{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
        auto b = tens{{3}, {14, 32, 50}};
        auto A_view = A.subview(slice{0, 2}, slice{0, 3});
        auto b_view = b.subview(slice{0, 3});
        solve_lls(A_view, b_view);
        CHECK(b[0] == doctest::Approx(15.6666).epsilon(0.01));
        CHECK(b[1] == doctest::Approx(6).epsilon(0.01));
    }
}

TEST_CASE("Matrix inversion with transposed tensors and views") {
    SUBCASE("Invert transposed fixed tensor") {
        mat2 A{{1, 2, 3, 4}};
        auto At = A.transpose();
        auto At_inv = At.inv();
        auto I = At * At_inv;
        CHECK(I[0, 0] == doctest::Approx(1.0));
        CHECK(I[0, 1] == doctest::Approx(0.0));
        CHECK(I[1, 0] == doctest::Approx(0.0));
        CHECK(I[1, 1] == doctest::Approx(1.0));
    }

    SUBCASE("Invert transposed dynamic tensor") {
        auto A = tens{{2, 2}, {1, 2, 3, 4}};
        auto At = A.transpose();
        auto At_inv = At.inv();
        auto I = At * At_inv;
        CHECK(I[0, 0] == doctest::Approx(1.0));
        CHECK(I[0, 1] == doctest::Approx(0.0));
        CHECK(I[1, 0] == doctest::Approx(0.0));
        CHECK(I[1, 1] == doctest::Approx(1.0));
    }

    SUBCASE("Invert fixed tensor view") {
        mat3 A{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        auto A_view = A.subview<2, 2>(slice{0, 2}, slice{0, 2});
        auto A_view_inv = A_view.inv();
        auto I = A_view * A_view_inv;
        CHECK(I[0, 0] == doctest::Approx(1.0));
        CHECK(I[0, 1] == doctest::Approx(0.0));
        CHECK(I[1, 0] == doctest::Approx(0.0));
        CHECK(I[1, 1] == doctest::Approx(1.0));
    }

    SUBCASE("Invert dynamic tensor view") {
        auto A = tens{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
        auto A_view = A.subview(slice{0, 2}, slice{0, 2});
        auto A_view_inv = A_view.inv();
        auto I = A_view * A_view_inv;
        CHECK(I[0, 0] == doctest::Approx(1.0));
        CHECK(I[0, 1] == doctest::Approx(0.0));
        CHECK(I[1, 0] == doctest::Approx(0.0));
        CHECK(I[1, 1] == doctest::Approx(1.0));
    }
}

TEST_CASE("Solve linear least squares - explicit overdetermined and underdetermined cases") {
    SUBCASE("Overdetermined system with fixed tensors") {
        // More equations than unknowns
        mat4x3 A{{1, 1, 1, 2, 3, 4, 1, 2, 3, 2, 1, 3}};
        vec4 b{{1, 2, 3, 4}};

        solve_lls(A, b);

        // Expected solution
        CHECK(b[0] == doctest::Approx(3.5384615).epsilon(0.0001));
        CHECK(b[1] == doctest::Approx(0.179487).epsilon(0.0001));
        CHECK(b[2] == doctest::Approx(-1.07692).epsilon(0.0001));
    }

    SUBCASE("Underdetermined system with fixed tensors") {
        // Fewer equations than unknowns
        mat3x4 A{{1, 1, 1, 1, 2, 3, 4, 5, 3, 5, 2, 4}};
        vec4 b{{10, 20, 30, 0}};

        solve_lls(A, b);

        // Expected solution (minimum norm solution)
        CHECK(b[0] == doctest::Approx(0.992998).epsilon(0.0001));
        CHECK(b[1] == doctest::Approx(9.89815).epsilon(0.0001));
        CHECK(b[2] == doctest::Approx(-0.127307).epsilon(0.0001));
        CHECK(b[3] == doctest::Approx(-0.0763853).epsilon(0.0001));
    }

    SUBCASE("Overdetermined system with dynamic tensors") {
        auto A = dtens{{4, 3}, {1, 1, 1, 2, 3, 4, 1, 2, 3, 2, 1, 3}};
        auto b = dtens{{4}, {1, 2, 3, 4}};

        solve_lls(A, b);

        // Check shape of result
        CHECK(b.size() == 4);

        // Expected solution
        CHECK(b[0] == doctest::Approx(3.5384615).epsilon(0.0001));
        CHECK(b[1] == doctest::Approx(0.179487).epsilon(0.0001));
        CHECK(b[2] == doctest::Approx(-1.07692).epsilon(0.0001));
    }

    SUBCASE("Underdetermined system with dynamic tensors") {
        auto A = dtens{{3, 4}, {1, 1, 1, 1, 2, 3, 4, 5, 3, 5, 2, 4}};
        auto b = dtens{{4}, {10, 20, 30, 0}};

        solve_lls(A, b);

        // Check shape of result
        CHECK(b.size() == 4);

        // Expected solution (minimum norm solution)
        CHECK(b[0] == doctest::Approx(0.992998).epsilon(0.0001));
        CHECK(b[1] == doctest::Approx(9.89815).epsilon(0.0001));
        CHECK(b[2] == doctest::Approx(-0.127307).epsilon(0.0001));
        CHECK(b[3] == doctest::Approx(-0.0763853).epsilon(0.0001));
    }
}

TEST_CASE("Matrix operations with quantity types") {
    using namespace squint::units;
    SUBCASE("Matrix multiplication with length and time") {
        mat2_t<length> A{{length::meters(1), length::meters(2), length::meters(3), length::meters(4)}};
        mat2_t<squint::units::time> B{{time::seconds(2), time::seconds(1), time::seconds(4), time::seconds(3)}};

        auto C = A * B;

        static_assert(
            std::is_same_v<decltype(C), mat2_t<quantity<float, mult_t<dimensions::length, dimensions::time>>>>,
            "Result should be a matrix of length * time");

        CHECK(C[0, 0].value() == doctest::Approx(5));
        CHECK(C[0, 1].value() == doctest::Approx(13));
        CHECK(C[1, 0].value() == doctest::Approx(8));
        CHECK(C[1, 1].value() == doctest::Approx(20));
    }

    SUBCASE("Matrix-vector multiplication with velocity and mass") {
        mat2_t<velocity> A{{velocity::meters_per_second(1), velocity::meters_per_second(2),
                            velocity::meters_per_second(3), velocity::meters_per_second(4)}};
        vec2_t<mass> v{{mass::kilograms(2), mass::kilograms(1)}};

        auto result = A * v;

        static_assert(std::convertible_to<decltype(result)::value_type,
                                          quantity<float, mult_t<dimensions::velocity, dimensions::mass>>>,
                      "Result should be a vector of velocity * mass (momentum)");

        CHECK(result[0].value() == doctest::Approx(5));
        CHECK(result[1].value() == doctest::Approx(8));
    }

    SUBCASE("Matrix addition with force") {
        mat2_t<force> A{{force::newtons(1), force::newtons(2), force::newtons(3), force::newtons(4)}};
        mat2_t<force> B{{force::newtons(5), force::newtons(6), force::newtons(7), force::newtons(8)}};

        auto C = A + B;

        static_assert(std::is_same_v<decltype(C), mat2_t<force>>, "Result should be a matrix of force");

        CHECK(C[0, 0].value() == doctest::Approx(6));
        CHECK(C[0, 1].value() == doctest::Approx(10));
        CHECK(C[1, 0].value() == doctest::Approx(8));
        CHECK(C[1, 1].value() == doctest::Approx(12));
    }

    SUBCASE("Matrix-scalar multiplication with energy") {
        mat2_t<energy> A{{energy::joules(1), energy::joules(2), energy::joules(3), energy::joules(4)}};
        auto scalar = dimensionless(2);

        auto B = A * scalar;

        static_assert(std::convertible_to<decltype(B)::value_type, energy>, "Result should be a matrix of energy");

        CHECK(B[0, 0].value() == doctest::Approx(2));
        CHECK(B[0, 1].value() == doctest::Approx(6));
        CHECK(B[1, 0].value() == doctest::Approx(4));
        CHECK(B[1, 1].value() == doctest::Approx(8));
    }

    SUBCASE("Matrix inverse with dimensionless quantities") {
        mat2_t<dimensionless> A{{dimensionless(1), dimensionless(2), dimensionless(3), dimensionless(4)}};

        auto A_inv = A.inv();

        static_assert(std::convertible_to<decltype(A_inv)::value_type, dimensionless>,
                      "Result should be a matrix of dimensionless quantities");

        CHECK(A_inv[0, 0].value() == doctest::Approx(-2));
        CHECK(A_inv[0, 1].value() == doctest::Approx(1.5));
        CHECK(A_inv[1, 0].value() == doctest::Approx(1));
        CHECK(A_inv[1, 1].value() == doctest::Approx(-0.5));
    }

    SUBCASE("Solving linear system with pressure and volume") {
        mat2_t<pressure> A{{pressure::pascals(3), pressure::pascals(2), pressure::pascals(1), pressure::pascals(1)}};
        vec2_t<volume> b{{volume::cubic_meters(7), volume::cubic_meters(3)}};

        auto x = b / A;

        static_assert(std::convertible_to<decltype(x)::value_type,
                                          quantity<float, squint::div_t<dimensions::volume, dimensions::pressure>>>,
                      "Result should be a vector of volume / pressure");

        CHECK(x[0].value() == doctest::Approx(4));
        CHECK(x[1].value() == doctest::Approx(-5));
    }
}