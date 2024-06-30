#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "squint/linalg.hpp"
#include "squint/optimize.hpp"

// TODO: add tests for matrix mult and inverse of transposed and sparse matrix refs
using namespace squint;
using namespace squint::quantities;
TEST_CASE("Fixed tensors construction and initialization", "[fixed construction]") {
    SECTION("construct tensors and scalars") {
        // tensors
        dmat4 t0;                                                 // default constructor
        dmat4 t1{};                                               // default constructor
        dmat4 t2(1);                                              // fill with value
        dmat4 t3(t2[0][0]);                                       // fill with scalar ref
        dmat4 t4(dscalar(1));                                     // fill with scalar
        dmat4 t5(t4[0]);                                          // fill with tensor_ref
        dmat4 t6(dvec4(1));                                       // fill with tensor
        dmat4 t7{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // list constructor with values
        dmat4 t8{t7[0], t7[1], t7[2], t7[3]};                     // list constructor with tensor_refs
        dmat4 t9{dvec4(1), dvec4(1), dvec4(1), dvec4(1)};         // list constructor with tensors
        dmat4 t10(t9);                                            // copy construction tensor
        dmat4 t11 = t10;                                          // copy initialization tensor (uses copy constructor)
        dvec4 t12(t11.at<4>(0, 0));                               // copy construction tensor_ref
        dvec4 t13 = t11.at<4>(0, 0); // copy initialization tensor_ref (uses copy constructor)
        dmat4 t14 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // list initialization with values
        dmat4 t15 = {t7[0], t7[1], t7[2], t7[3]};                     // list initialization with tensor_refs
        dmat4 t16 = {dvec4(1), dvec4(1), dvec4(1), dvec4(1)};         // list initialization with tensors
        // scalars
        dscalar s0{};              // default constructor
        dscalar s1;                // default constructor
        dscalar s2(1);             // copy constructor value
        dscalar s3(s2);            // copy constructor scalar
        dscalar s4(t11[0][0]);     // copy constructor scalar_ref
        dscalar s5 = 1;            // copy initalization value
        dscalar s6 = s5;           // copy initalization scalar
        dscalar s7 = t11[0][0];    // copy initalization scalar_ref
        dscalar s8{1};             // list constructor value
        dscalar s9{s8};            // list constructor scalar
        dscalar s10{t11[0][0]};    // list constructor scalar_ref
        dscalar s11 = {1};         // list initalization value
        dscalar s12 = {s7};        // list initalization scalar
        dscalar s13 = {t11[0][0]}; // list initalization scalar_ref
        // check values are initalized to 1 or 0 if default constructed
        for (const auto &t : {t0, t1}) {
            for (int i = 0; i < 16; i++) {
                REQUIRE(t.data()[i] == 0); // Tensors must be default constructed to zero
            }
        }
        for (const auto &t : {t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t14, t15, t16}) {
            for (int i = 0; i < 16; i++) {
                REQUIRE(t.data()[i] == 1); // Tensors must be initalized to one
            }
        }
        for (const auto &t : {t12, t13}) {
            for (int i = 0; i < 4; i++) {
                REQUIRE(t.data()[i] == 1); // Tensors must be initalized to one
            }
        }
        for (const auto &s : {s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13}) {
            REQUIRE(s.data()[0] == 1); // Scalars must be initalized to one
        }
        for (const auto &s : {s0, s1}) {
            REQUIRE(s.data()[0] == 0); // Scalars must be default constructed to zero
        }
    }
    SECTION("implicit conversion to underlying type") {
        // test implicit conversions
        dmat4 t(1);
        double d = dscalar(2);
        REQUIRE(d == 2);
        double &dref = t[0][0];
        dref = 2;
        REQUIRE(t[0][0] == 2);
    }
}

TEST_CASE("Fixed tensors assingment indexing and iterators", "[fixed assignment and indexing]") {
    dmat4 A;
    dmat4 B(1);
    dvec4 C;
    A = B;             // copy assignment tensor to tensor
    C = A.at<4>(0, 0); // copy assignment tensor_ref to tensor
    A[0] = drvec4(2);  // copy assignment tensor to tensor_ref
    A[1] = A[0];       // copy assignment tensor_ref to tensor_ref
    SECTION("copy assignment") {
        for (const auto &elem : A.at<2, 4>(0, 0)) {
            REQUIRE(elem == 2); // Copy assignment should assign 2 here
        }
        for (const auto &elem : A.at<2, 4>(2, 0)) {
            REQUIRE(elem == 1); // Copy assignment should assign 1 here
        }
    }
    SECTION("indexing assignment") {
        // const and non const index operator
        const dmat4 D = A;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                B[i][j] = D[i][j];
            }
        }
        int i = 0;
        for (const auto &block : B.block<2, 4>()) {
            for (const auto &elem : block) {
                if (i == 0) {
                    REQUIRE(elem == 2); // Indexing should assign 2 here
                } else {
                    REQUIRE(elem == 1); // Indexing should assign 1 here
                }
            }
            i++;
        }
    }
    SECTION("scalar assignment") {
        A[0][0] = 3;          // scalar ref assingment from value
        A[1][0] = A[0][0];    // scalar ref assingment from scalar ref
        A[2][0] = dscalar(3); // scalar ref assingment from scalar
        dscalar s1;
        dscalar s2;
        dscalar s3;
        s1 = 3;          // scalar assingment from value
        s2 = A[0][0];    // scalar assingment from scalar ref
        s3 = dscalar(3); // scalar assingment from scalar
        REQUIRE(A[0][0].data()[0] == 3);
        REQUIRE(A[1][0].data()[0] == 3);
        REQUIRE(A[2][0].data()[0] == 3);
        REQUIRE(s1.data()[0] == 3);
        REQUIRE(s2.data()[0] == 3);
        REQUIRE(s3.data()[0] == 3);
    }
}

TEST_CASE("Fixed reshaping and printing", "[fixed printing]") {
    // second order tensors print columns as rows
    // first order tensors print columns as rows
    std::stringstream ss;
    dmat4 A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const dmat4 B = A;
    ss << A;
    REQUIRE(ss.str() == "{{1, 2, 3, 4},\n{5, 6, 7, 8},\n{9, 10, 11, 12},\n{13, 14, 15, 16}}");
    ss.str(std::string());
    ss << A[1];
    REQUIRE(ss.str() == "{2, 6, 10, 14}");
    ss.str(std::string());
    ss << A[2][3];
    REQUIRE(ss.str() == "15");
    ss.str(std::string());
    auto K = A.as_shape<2, 8>();
    ss << K;
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << B.as_shape<2, 8>();
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << A.transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << A.transpose<2, 1, 0>();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.transpose<2, 1, 0>();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    auto L = K.transpose();
    ss << L;
    REQUIRE(ss.str() == "{{1, 3, 5, 7, 9, 11, 13, 15},\n{2, 4, 6, 8, 10, 12, 14, 16}}");
    ss.str(std::string());
    ss << L.transpose<2, 1, 0>();
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << B.as_ref().transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.as_ref().transpose<2, 1, 0>();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << A.flatten();
    REQUIRE(ss.str() == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}");
    ss.str(std::string());
    ss << B.flatten();
    REQUIRE(ss.str() == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}");
    ss.str(std::string());
}

TEST_CASE("Fixed math and comparisons", "[fixed math]") {
    dscalar s1(1);
    dscalar s2(1);
    dscalar s3(2);
    SECTION("scalar comparisons") {
        REQUIRE(s1 == s2);
        REQUIRE(s1 == 1);
        REQUIRE(1 == s1);
        REQUIRE(s2 < s3);
        REQUIRE(s2 < 2);
        REQUIRE(1 < s3);
        REQUIRE(s3 > s2);
        REQUIRE(s3 > 1);
        REQUIRE(2 > s2);
        REQUIRE(s1 >= s2);
        REQUIRE(s1 >= 1);
        REQUIRE(1 >= s2);
        REQUIRE(s1 <= s2);
        REQUIRE(s1 <= 1);
        REQUIRE(1 <= s2);
        REQUIRE(s3 != s2);
        REQUIRE(s3 != 1);
        REQUIRE(2 != s2);
    }
    dmat4 A(1);
    dmat4 B = A;
    drvec4 C(1);
    SECTION("tensor comparisons") {
        REQUIRE(A == B);
        REQUIRE(C == A[0]);
        REQUIRE(A[0] == C);
        B[2][3] = 2;
        REQUIRE(A != B);
        REQUIRE(C != B[2]);
        REQUIRE(B[2] != C);
        dscalar s4 = 1 + 5e-16;
        REQUIRE(approx_equal(s1, s4));
        REQUIRE(s1 != s4);
        REQUIRE(approx_equal(s2, 1 + 5e-16));
        REQUIRE(approx_equal(1 + 5e-16, s2));
        REQUIRE(s2 != 1 + 5e-16);
        B[2][3] = 1 + 2 * 5e-16;
        REQUIRE(approx_equal(A, B));
        REQUIRE(A != B);
    }
    SECTION("scalar compound assignment operators") {
        s1 += 1;
        s1 += s2;
        s1 += A[0][0];
        double d1 = 1;
        d1 += s2;
        d1 += 1;
        d1 += A[0][0];
        REQUIRE(s1 == d1);
        s1 -= 1;
        s1 -= s2;
        s1 -= A[0][0];
        d1 -= s2;
        d1 -= 1;
        d1 -= A[0][0];
        REQUIRE(s1 == d1);
        B[2][3] = 2;
        s1 *= 2;
        s1 *= s1;
        s1 *= B[2][3];
        d1 *= s1;
        d1 -= 6;
        d1 *= 2;
        d1 *= B[2][3];
        REQUIRE(s1 == d1);
        s1 /= 2;
        s1 /= s1;
        s1 /= B[2][3];
        d1 /= 2;
        d1 /= dscalar(4);
        d1 /= B[2][3];
        REQUIRE(s1 == d1);
    }
    dmat4 D(1);
    dmat4 E(1);
    dmat4 F(1);
    dmat4 G(2);
    drvec4 H(1);
    SECTION("tensor compound assignment operators") {
        E += D;
        REQUIRE(E == dmat4(2));
        C += E[0];
        REQUIRE(C == drvec4(3));
        E[0] += C;
        REQUIRE(E[0] == drvec4(5));
        E[0] += E[0];
        REQUIRE(E[0] == drvec4(10));
        F -= G;
        REQUIRE(F == dmat4(-1));
        H -= F[0];
        REQUIRE(H == drvec4(2));
        G[0] -= H;
        REQUIRE(G[0] == drvec4(0));
        G[0] -= F[0];
        REQUIRE(G[0] == drvec4(1));
        G[0] *= 2;
        G[0] *= G[0][0];
        G[0] *= dscalar(2);
        G *= 2;
        G *= G[0][0];
        G *= dscalar(2);
        REQUIRE(G[0] == drvec4(512));
        G[0] /= 2;
        G[0] /= G[0][0];
        G[0] /= dscalar(2);
        G /= 2;
        G /= G[0][0];
        G /= dscalar(2);
        REQUIRE(G[0] == drvec4(0.5));
    }
    dscalar s5(1);
    dscalar s6(2);
    G[0][0] = 0.5;
    SECTION("scalar operations") {
        auto s7 = s5 + s6;
        REQUIRE(s7 == 3);
        auto s8 = s5 + G[0][0];
        REQUIRE(s8 == 1.5);
        auto s9 = s5 + 1;
        REQUIRE(s9 == 2);
        auto s10 = G[0][0] + s5;
        REQUIRE(s10 == 1.5);
        auto s11 = G[0][0] + G[0][0];
        REQUIRE(s11 == 1);
        auto s12 = G[0][0] + 1;
        REQUIRE(s12 == 1.5);
        auto s13 = 1 + s6;
        REQUIRE(s13 == 3);
        auto s14 = 1 + G[0][0];
        REQUIRE(s14 == 1.5);
        s7 = s5 - s6;
        REQUIRE(s7 == -1);
        s8 = s5 - G[0][0];
        REQUIRE(s8 == 0.5);
        s9 = s5 - 1;
        REQUIRE(s9 == 0);
        s10 = G[0][0] - s5;
        REQUIRE(s10 == -0.5);
        s11 = G[0][0] - G[0][0];
        REQUIRE(s11 == 0);
        s12 = G[0][0] - 1;
        REQUIRE(s12 == -0.5);
        s13 = 1 - s6;
        REQUIRE(s13 == -1);
        s14 = 1 - G[0][0];
        REQUIRE(s14 == 0.5);
        s7 = s5 * s6;
        REQUIRE(s7 == 2);
        s8 = s5 * G[0][0];
        REQUIRE(s8 == 0.5);
        s9 = s5 * 1;
        REQUIRE(s9 == 1);
        s10 = G[0][0] * s5;
        REQUIRE(s10 == 0.5);
        s11 = G[0][0] * G[0][0];
        REQUIRE(s11 == 0.25);
        s12 = G[0][0] * 1;
        REQUIRE(s12 == 0.5);
        s13 = 1 * s6;
        REQUIRE(s13 == 2);
        s14 = 1 * G[0][0];
        REQUIRE(s14 == 0.5);
        s7 = s5 / s6;
        REQUIRE(s7 == 0.5);
        s8 = s5 / G[0][0];
        REQUIRE(s8 == 2);
        s9 = s5 / 1;
        REQUIRE(s9 == 1);
        s10 = G[0][0] / s5;
        REQUIRE(s10 == 0.5);
        s11 = G[0][0] / G[0][0];
        REQUIRE(s11 == 1);
        s12 = G[0][0] / 1;
        REQUIRE(s12 == 0.5);
        s13 = 1 / s6;
        REQUIRE(s13 == 0.5);
        s14 = 1 / G[0][0];
        REQUIRE(s14 == 2);
    }
    drvec4 t1(1);
    drvec4 t2(2);
    dmat4 t3(2);
    drvec4 row_vec(5);
    dvec4 col_vec(5);
    dmat4 mat(2);
    dtensor<4, 4, 4> tens(2);
    SECTION("tensor operations") {
        auto t4 = t1 + t2;
        auto t5 = t1 + t3[0];
        auto t6 = t3[0] + t1;
        auto t7 = t3[0] + t3[0];
        REQUIRE(t4 == drvec4(3));
        REQUIRE(t5 == drvec4(3));
        REQUIRE(t6 == drvec4(3));
        REQUIRE(t7 == drvec4(4));
        t4 = t1 - t2;
        t5 = t1 - t3[0];
        t6 = t3[0] - t1;
        t7 = t3[0] - t3[0];
        REQUIRE(t4 == drvec4(-1));
        REQUIRE(t5 == drvec4(-1));
        REQUIRE(t6 == drvec4(1));
        REQUIRE(t7 == drvec4(0));
        double s14 = 2;
        auto s15 = -s14;
        auto s16 = -t3[0][0];
        auto t8 = -t3[0];
        auto t9 = -t3;
        REQUIRE(s15 == -2);
        REQUIRE(s16 == -2);
        REQUIRE(t8 == drvec4(-2));
        REQUIRE(t9 == dmat4(-2));
        auto r1 = row_vec * mat;
        auto r1r = row_vec * tens[0].simplify_shape();
        auto r1rr = col_vec.transpose() * tens[0].simplify_shape();
        auto r1rrr = col_vec.transpose() * mat;
        REQUIRE(r1 == drvec4(40));
        REQUIRE(r1r == drvec4(40));
        REQUIRE(r1rr == drvec4(40));
        REQUIRE(r1rrr == drvec4(40));
        auto r2 = mat * col_vec;
        auto r2r = mat * row_vec.transpose();
        auto r2rr = mat.as_shape<4, 4>() * col_vec;
        auto r2rrr = mat.as_shape<4, 4>() * col_vec.as_shape<4>();
        REQUIRE(r2 == dvec4(40));
        REQUIRE(r2r == dvec4(40));
        REQUIRE(r2rr == dvec4(40));
        REQUIRE(r2rrr == dvec4(40));
        auto r3 = row_vec * col_vec;
        auto r3r = row_vec.as_ref() * col_vec;
        auto r3rr = row_vec.as_ref() * col_vec.as_ref();
        auto r3rrr = row_vec * col_vec.as_ref();
        REQUIRE(r3 == 100);
        REQUIRE(r3r == 100);
        REQUIRE(r3rr == 100);
        REQUIRE(r3rrr == 100);
        auto r4 = col_vec * row_vec;
        auto r4r = col_vec.as_ref() * row_vec;
        auto r4rr = col_vec.as_ref() * row_vec.as_ref();
        auto r4rrr = col_vec * row_vec.as_ref();
        REQUIRE(r4 == dmat4(25));
        REQUIRE(r4r == dmat4(25));
        REQUIRE(r4rr == dmat4(25));
        REQUIRE(r4rrr == dmat4(25));
        auto r5 = mat * mat;
        auto r5r = mat.transpose() * mat;
        auto r5rr = mat.transpose() * mat.transpose();
        auto r5rrr = mat * mat.transpose();
        REQUIRE(r5 == dmat4(16));
        REQUIRE(r5r == dmat4(16));
        REQUIRE(r5rr == dmat4(16));
        REQUIRE(r5rrr == dmat4(16));
        auto r6 = mat * 2;
        REQUIRE(r6 == dmat4(4));
        auto r7 = mat * dscalar(2);
        REQUIRE(r7 == dmat4(4));
        auto r8 = mat * mat[0][0];
        REQUIRE(r8 == dmat4(4));
        auto r9 = mat.as_ref() * 2;
        REQUIRE(r9 == dmat4(4));
        auto r10 = mat.as_ref() * dscalar(2);
        REQUIRE(r10 == dmat4(4));
        auto r11 = mat.as_ref() * mat[0][0];
        REQUIRE(r11 == dmat4(4));
    }
    dmat2 AA{4, 2, 7, 6};
    dmat2 LL = AA;
    dvec2 bb{3, 4};
    SECTION("functions") {
        auto r12 = dot(row_vec, col_vec);
        auto r13 = dot(row_vec, col_vec.as_ref());
        auto r14 = dot(row_vec.as_ref(), col_vec.as_ref());
        auto r15 = dot(row_vec, col_vec.as_ref());
        REQUIRE(r12 == 100);
        REQUIRE(r13 == 100);
        REQUIRE(r14 == 100);
        REQUIRE(r15 == 100);
        dvec3 u{2, 1, 2};
        dvec3 v{3, 3, 3};
        auto r16 = cross(u, v);
        auto r17 = cross(u.as_ref(), v);
        auto r18 = cross(u.as_ref(), v.as_ref());
        auto r19 = cross(u, v.as_ref());
        REQUIRE(r16 == dvec3({-3, 0, 3}));
        REQUIRE(r17 == dvec3({-3, 0, 3}));
        REQUIRE(r18 == dvec3({-3, 0, 3}));
        REQUIRE(r19 == dvec3({-3, 0, 3}));
        auto n1 = norm(u);
        auto n2 = norm(u.as_ref());
        REQUIRE(n1 == 3);
        REQUIRE(n2 == 3);
        auto xx = solve(AA, bb);
        REQUIRE(xx == dvec2({-1, 1}));
        auto lls_result1 = solve_lls(AA, bb);
        REQUIRE(approx_equal(lls_result1, xx));
        auto lls_result2 = bb / AA;
        REQUIRE(lls_result1 == lls_result2);
        dmat<2, 3> A_under{1, -1, 1, -1, 1, 1};
        dvec2 b_under{1, 0};
        auto x_under = solve_lls(A_under, b_under);
        REQUIRE(approx_equal(x_under, dvec3({1. / 4., 1. / 4., 1. / 2.})));
        drvec2 a_under_vec{1, 2};
        dscalar b_under_scalar{2.25};
        auto x_under_vec = solve_lls(a_under_vec, b_under_scalar);
        REQUIRE(approx_equal(x_under_vec, dvec2({0.45, 0.9})));
        dmat<3, 2> A_over{0., 1., 0., 1.1, 0., -0.2};
        dvec3 b_over{1.1, -1.1, -0.2};
        auto x_over = solve_lls(A_over, b_over);
        REQUIRE(approx_equal(x_over, dvec2({-1.1, 1.})));
        dvec4 a_over_vec{1, 2, 8, 5};
        dvec4 b_over_vec{3, 4, 7, 8};
        dmat4 B_over{3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8};
        auto x_over_vec = solve_lls(a_over_vec, b_over_vec);
        auto X_over_vec = solve_lls(a_over_vec, B_over);
        double over_result = 1.13829787234;
        REQUIRE(approx_equal(x_over_vec, over_result));
        REQUIRE(approx_equal(X_over_vec, drvec4(over_result)));
        auto LL_inv = inv(LL);
        REQUIRE(approx_equal(LL_inv, dmat2({0.6, -0.2, -0.7, 0.4})));
        auto LL_pinv = pinv(LL);
        REQUIRE(approx_equal(LL_pinv, dmat2({0.6, -0.2, -0.7, 0.4})));
    }
    dmat4 CC;
    SECTION("row and column iterators") {
        for (auto &row : CC.rows()) {
            REQUIRE(approx_equal(row, drvec4(0.)));
        }
        for (auto &row : bb.rows()) {
            REQUIRE(row.size() == 1);
        }
        for (auto &row : row_vec.rows()) {
            REQUIRE(row.size() == 4);
            REQUIRE(row.shape(0) == 1);
            REQUIRE(row.shape(1) == 4);
        }
        for (auto &col : CC.cols()) {
            REQUIRE(col.size() == 4);
            REQUIRE(col.shape(0) == 4);
            REQUIRE(col.shape(1) == 1);
        }
        for (auto &col : bb.cols()) {
            REQUIRE(col.size() == 2);
            REQUIRE(col.shape(0) == 2);
            REQUIRE(col.shape(1) == 1);
        }
        for (auto &col : row_vec.cols()) {
            REQUIRE(col.size() == 1);
        }
        for (const auto &row : CC.rows()) {
            REQUIRE(approx_equal(row, drvec4(0.)));
        }
        for (const auto &row : bb.rows()) {
            REQUIRE(row.size() == 1);
        }
        for (const auto &row : row_vec.rows()) {
            REQUIRE(row.size() == 4);
            REQUIRE(row.shape(0) == 1);
            REQUIRE(row.shape(1) == 4);
        }
        for (const auto &col : CC.cols()) {
            REQUIRE(col.size() == 4);
            REQUIRE(col.shape(0) == 4);
            REQUIRE(col.shape(1) == 1);
        }
        for (const auto &col : bb.cols()) {
            REQUIRE(col.size() == 2);
            REQUIRE(col.shape(0) == 2);
            REQUIRE(col.shape(1) == 1);
        }
        for (const auto &col : row_vec.cols()) {
            REQUIRE(col.size() == 1);
        }
    }
}

TEST_CASE("Dynamic tensors construction and initialization", "[dynamic construction]") {
    SECTION("construct tensors and scalars") {
        // tensors
        dtens t0;                                                 // default constructor
        dtens t1{};                                               // default constructor
        dtens t2({4, 4}, 1);                                      // fill with value
        dtens t3({4, 4}, t2[0][0]);                               // fill with scalar ref
        dtens t4({4, 4}, dscalar(1));                             // fill with scalar
        dtens t5({4, 4}, t4[0]);                                  // fill with tensor_ref
        dtens t6({4, 4}, dtens({4}, 1));                          // fill with tensor
        dtens t7{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // list constructor with values
        t7.reshape({4, 4});
        dtens t8{t7[0], t7[1], t7[2], t7[3]};                                 // list constructor with tensor_refs
        dtens t9{dtens({4}, 1), dtens({4}, 1), dtens({4}, 1), dtens({4}, 1)}; // list constructor with tensors
        dtens t10(t9);                                                        // copy construction tensor
        dtens t11 = t10;    // copy initialization tensor (uses copy constructor)
        dtens t12(t11[0]);  // copy construction tensor_ref
        dtens t13 = t11[0]; // copy initialization tensor_ref (uses copy constructor)
        dtens t14 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // list initialization with values
        dtens t15 = {t7[0], t7[1], t7[2], t7[3]};                     // list initialization with tensor_refs
        dtens t16 = {dtens({4}, 1), dtens({4}, 1), dtens({4}, 1), dtens({4}, 1)}; // list initialization with tensors
        dtens t17({1, 4}, t11[0]);                                                // fill with one block
        dtens t18({1, 4});                                                        // construct uninitialized tensor
        // check iteration over default construction does not segfault
        int i = 0;
        for (const auto &t : {t0, t1}) {
            for (const auto &elem : t) {
                i++;
            }
            REQUIRE(i == 0); // Default constructed tensor should be empty
        }
        for (const auto block : t1.block({})) {
            i++;
        }
        REQUIRE(i == 0); // Default constructed tensor should be empty
        for (const auto &t : {t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t14, t15, t16}) {
            for (int i = 0; i < 16; i++) {
                REQUIRE(t.data()[i] == 1.); // tensors should be initalized to one
            }
        }
        for (const auto &t : {t12, t13, t17}) {
            for (int i = 0; i < 4; i++) {
                REQUIRE(t.data()[i] == 1.); // tensors should be initalized to one
            }
        }
    }
    SECTION("implicit conversion to underlying type") {
        // test implicit conversions
        dtens t({4, 4}, 1);
        double d = dscalar(2);
        REQUIRE(d == 2);
        double &dref = t[0][0];
        dref = 2;
        REQUIRE((t[0][0] == 2));
    }
}

TEST_CASE("Dynamic tensors assingment indexing and iterators", "[dynamic assignment and indexing]") {
    dtens A;
    dtens B({4, 4}, 1);
    dtens C;
    A = B;                   // copy assignment tensor to tensor
    C = A[0];                // copy assignment tensor_ref to tensor
    A[0] = dtens({1, 4}, 2); // copy assignment tensor to tensor_ref
    A[1] = A[0];             // copy assignment tensor_ref to tensor_ref
    SECTION("copy assignment") {
        for (const auto &elem : A.at({2, 4}, {0, 0})) {
            REQUIRE(elem == 2); // copy assignment should set elems to 2
        }
        for (const auto &elem : A.at({2, 4}, {2, 0})) {
            REQUIRE(elem == 1); // copy assignment should set elems to 1
        }
    }
    SECTION("indexing assignment") {
        // const and non const index operator
        const dtens D = A;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                B[i][j] = D[i][j];
            }
        }
        int i = 0;
        for (const auto &block : B.block({2, 4})) {
            for (const auto &elem : block) {
                if (i == 0) {
                    REQUIRE(elem == 2); // index assignment should set elems to 2
                } else {
                    REQUIRE(elem == 1); // index assignment should set elems to 1
                }
            }
            i++;
        }
    }
    SECTION("scalar assignment") {
        A[0][0] = 3;          // scalar ref assingment from value
        A[1][0] = A[0][0];    // scalar ref assingment from scalar ref
        A[2][0] = dscalar(3); // scalar ref assingment from scalar
        REQUIRE(A[0][0].data()[0] == 3);
        REQUIRE(A[1][0].data()[0] == 3);
        REQUIRE(A[2][0].data()[0] == 3);
    }
}

TEST_CASE("Dynamic reshaping and printing", "[dynamic printing]") {
    // need to visually inspect what is printed.
    // second order tensors print columns as rows
    // first order tensors print columns as rows
    // print empty tensor
    std::stringstream ss;
    dtens t0;   // default constructor
    dtens t1{}; // default constructor
    ss << t0 << t1;
    REQUIRE(ss.str().empty());
    dtens A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    A.reshape({4, 4});
    const dtens B = A;
    ss << A;
    REQUIRE(ss.str() == "{{1, 2, 3, 4},\n{5, 6, 7, 8},\n{9, 10, 11, 12},\n{13, 14, 15, 16}}");
    ss.str(std::string());
    ss << A[1];
    REQUIRE(ss.str() == "{2, 6, 10, 14}");
    ss.str(std::string());
    ss << A[2][3];
    REQUIRE(ss.str() == "15");
    ss.str(std::string());
    auto K = A.as_shape({2, 8});
    ss << K;
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << B.as_shape({2, 8});
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << A.transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << A.transpose({2, 1, 0});
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.transpose({2, 1, 0});
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    auto L = K.transpose();
    ss << L;
    REQUIRE(ss.str() == "{{1, 3, 5, 7, 9, 11, 13, 15},\n{2, 4, 6, 8, 10, 12, 14, 16}}");
    ss.str(std::string());
    ss << L.transpose({2, 1, 0});
    REQUIRE(ss.str() == "{{1, 2},\n{3, 4},\n{5, 6},\n{7, 8},\n{9, 10},\n{11, 12},\n{13, 14},\n{15, 16}}");
    ss.str(std::string());
    ss << B.as_ref().transpose();
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << B.as_ref().transpose({2, 1, 0});
    REQUIRE(ss.str() == "{{1, 5, 9, 13},\n{2, 6, 10, 14},\n{3, 7, 11, 15},\n{4, 8, 12, 16}}");
    ss.str(std::string());
    ss << A.flatten();
    REQUIRE(ss.str() == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}");
    ss.str(std::string());
    ss << B.flatten();
    REQUIRE(ss.str() == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}");
    ss.str(std::string());
}

TEST_CASE("Dynamic math and comparisons", "[dynamic math]") {
    dtens A({4, 4}, 1);
    dtens B = A;
    dtens C({1, 4}, 1);
    SECTION("comparisons") {
        REQUIRE(A == B);
        REQUIRE(C == A[0]);
        REQUIRE(A[0] == C);
        B[2][3] = 2;
        REQUIRE(A != B);
        REQUIRE(C != B[2]);
        REQUIRE(B[2] != C);
        B[2][3] = 1 + 2 * 5e-16;
        REQUIRE(approx_equal(A, B));
        REQUIRE(A != B);
        dtens D({4, 4}, 1);
        dtens E({4, 4}, 1);
        E += D;
        REQUIRE(E == dtens({4, 4}, 2));
        C += E[0];
        REQUIRE(C == dtens({1, 4}, 3));
        E[0] += C;
        REQUIRE(E[0] == dtens({1, 4}, 5));
        E[0] += E[0];
        REQUIRE(E[0] == dtens({1, 4}, 10));
    }
    dtens F({4, 4}, 1);
    dtens G({4, 4}, 2);
    dtens H({1, 4}, 1);
    SECTION("compound assignment operators") {
        F -= G;
        REQUIRE(F == dtens({4, 4}, -1));
        H -= F[0];
        REQUIRE(H == dtens({1, 4}, 2));
        G[0] -= H;
        REQUIRE(G[0] == dtens({1, 4}, 0));
        G[0] -= F[0];
        REQUIRE(G[0] == dtens({1, 4}, 1));
        G[0] *= 2.; // needs to be explicity a double to avoid possible ambiguity with built-in operator*=
        G[0] *= G[0][0];
        G[0] *= dscalar(2);
        G *= 2;
        G *= G[0][0];
        G *= dscalar(2);
        REQUIRE(G[0] == dtens({1, 4}, 512));
        G[0] /= 2.; // needs to be explicity a double to avoid possible ambiguity with built-in operator/=
        G[0] /= G[0][0];
        G[0] /= dscalar(2);
        G /= 2;
        G /= G[0][0];
        G /= dscalar(2);
        REQUIRE(G[0] == dtens({1, 4}, 0.5));
    }
    dscalar s5(1);
    dscalar s6(2);
    dtens row_vec({1, 4}, 5);
    dtens col_vec({4}, 5);
    dtens mat({4, 4}, 2);
    dtens tens({4, 4, 4}, 2);
    G[0][0] = 0.5;
    SECTION("operations") {
        auto s7 = s5 + s6;
        REQUIRE(s7 == 3);
        auto s8 = s5 + G[0][0];
        REQUIRE(s8 == 1.5);
        auto s9 = s5 + 1;
        REQUIRE(s9 == 2);
        auto s10 = G[0][0] + s5;
        REQUIRE(s10 == 1.5);
        auto s11 = G[0][0] + G[0][0];
        REQUIRE(s11 == 1);
        auto s12 = G[0][0] + 1;
        REQUIRE(s12 == 1.5);
        auto s13 = 1 + s6;
        REQUIRE(s13 == 3);
        auto s14 = 1 + G[0][0];
        REQUIRE(s14 == 1.5);
        s7 = s5 - s6;
        REQUIRE(s7 == -1);
        s8 = s5 - G[0][0];
        REQUIRE(s8 == 0.5);
        s9 = s5 - 1;
        REQUIRE(s9 == 0);
        s10 = G[0][0] - s5;
        REQUIRE(s10 == -0.5);
        s11 = G[0][0] - G[0][0];
        REQUIRE(s11 == 0);
        s12 = G[0][0] - 1;
        REQUIRE(s12 == -0.5);
        s13 = 1 - s6;
        REQUIRE(s13 == -1);
        s14 = 1 - G[0][0];
        REQUIRE(s14 == 0.5);
        s7 = s5 * s6;
        REQUIRE(s7 == 2);
        auto ppp = s5 * G[0][0];
        s8 = s5 * G[0][0];
        REQUIRE(s8 == 0.5);
        s9 = s5 * 1;
        REQUIRE(s9 == 1);
        s10 = G[0][0] * s5;
        REQUIRE(s10 == 0.5);
        s11 = G[0][0] * G[0][0];
        REQUIRE(s11 == 0.25);
        s12 = G[0][0] * 1;
        REQUIRE(s12 == 0.5);
        s13 = 1 * s6;
        REQUIRE(s13 == 2);
        s14 = 1 * G[0][0];
        REQUIRE(s14 == 0.5);
        s7 = s5 / s6;
        REQUIRE(s7 == 0.5);
        s8 = s5 / G[0][0];
        REQUIRE(s8 == 2);
        s9 = s5 / 1;
        REQUIRE(s9 == 1);
        s10 = G[0][0] / s5;
        REQUIRE(s10 == 0.5);
        s11 = G[0][0] / G[0][0];
        REQUIRE(s11 == 1);
        s12 = G[0][0] / 1;
        REQUIRE(s12 == 0.5);
        s13 = 1 / s6;
        REQUIRE(s13 == 0.5);
        s14 = 1 / G[0][0];
        REQUIRE(s14 == 2);
        dtens t1({1, 4}, 1);
        dtens t2({1, 4}, 2);
        dtens t3({4, 4}, 2);
        auto t4 = t1 + t2;
        auto t5 = t1 + t3[0];
        auto t6 = t3[0] + t1;
        auto t7 = t3[0] + t3[0];
        REQUIRE(t4 == dtens({1, 4}, 3));
        REQUIRE(t5 == dtens({1, 4}, 3));
        REQUIRE(t6 == dtens({1, 4}, 3));
        REQUIRE(t7 == dtens({1, 4}, 4));
        t4 = t1 - t2;
        t5 = t1 - t3[0];
        t6 = t3[0] - t1;
        t7 = t3[0] - t3[0];
        REQUIRE(t4 == dtens({1, 4}, -1));
        REQUIRE(t5 == dtens({1, 4}, -1));
        REQUIRE(t6 == dtens({1, 4}, 1));
        REQUIRE(t7 == dtens({1, 4}, 0));
        auto s15 = -s14;
        auto s16 = -t3[0][0];
        auto t8 = -t3[0];
        auto t9 = -t3;
        REQUIRE(s15 == -2);
        REQUIRE(s16 == -2);
        REQUIRE(t8 == dtens({1, 4}, -2));
        REQUIRE(t9 == dtens({4, 4}, -2));
        auto r1 = row_vec * mat;
        auto r1r = row_vec * tens[0].simplify_shape();
        auto r1rr = col_vec.transpose() * tens[0].simplify_shape();
        auto r1rrr = col_vec.transpose() * mat;
        REQUIRE(r1 == dtens({1, 4}, 40));
        REQUIRE(r1r == dtens({1, 4}, 40));
        REQUIRE(r1rr == dtens({1, 4}, 40));
        REQUIRE(r1rrr == dtens({1, 4}, 40));
        auto r2 = mat * col_vec;
        auto r2r = mat * row_vec.transpose();
        auto r2rr = mat.as_shape({4, 4}) * col_vec;
        auto r2rrr = mat.as_shape({4, 4}) * col_vec.as_shape({4});
        REQUIRE(r2 == dtens({4}, 40));
        REQUIRE(r2r == dtens({4}, 40));
        REQUIRE(r2rr == dtens({4}, 40));
        REQUIRE(r2rrr == dtens({4}, 40));
        auto r3 = row_vec * col_vec;
        auto r3r = row_vec.as_ref() * col_vec;
        auto r3rr = row_vec.as_ref() * col_vec.as_ref();
        auto r3rrr = row_vec * col_vec.as_ref();
        REQUIRE(r3 == 100);
        REQUIRE(r3r == 100);
        REQUIRE(r3rr == 100);
        REQUIRE(r3rrr == 100);
        auto r4 = col_vec * row_vec;
        auto r4r = col_vec.as_ref() * row_vec;
        auto r4rr = col_vec.as_ref() * row_vec.as_ref();
        auto r4rrr = col_vec * row_vec.as_ref();
        REQUIRE(r4 == dtens({4, 4}, 25));
        REQUIRE(r4r == dtens({4, 4}, 25));
        REQUIRE(r4rr == dtens({4, 4}, 25));
        REQUIRE(r4rrr == dtens({4, 4}, 25));
        auto r5 = mat * mat;
        auto r5r = mat.transpose() * mat;
        auto r5rr = mat.transpose() * mat.transpose();
        auto r5rrr = mat * mat.transpose();
        REQUIRE(r5 == dtens({4, 4}, 16));
        REQUIRE(r5r == dtens({4, 4}, 16));
        REQUIRE(r5rr == dtens({4, 4}, 16));
        REQUIRE(r5rrr == dtens({4, 4}, 16));
        auto r6 = mat * 2;
        REQUIRE(r6 == dtens({4, 4}, 4));
        auto r7 = mat * dscalar(2);
        REQUIRE(r7 == dtens({4, 4}, 4));
        auto r8 = mat * mat[0][0];
        REQUIRE(r8 == dtens({4, 4}, 4));
        auto r9 = mat.as_ref() * 2;
        REQUIRE(r9 == dtens({4, 4}, 4));
        auto r10 = mat.as_ref() * dscalar(2);
        REQUIRE(r10 == dtens({4, 4}, 4));
        auto r11 = mat.as_ref() * mat[0][0];
        REQUIRE(r11 == dtens({4, 4}, 4));
    }
    dtens AA{4, 2, 7, 6};
    AA.reshape({2, 2});
    dtens LL = AA;
    dtens bb{3, 4};
    SECTION("functions") {
        auto r12 = dot(row_vec, col_vec);
        auto r13 = dot(row_vec, col_vec.as_ref());
        auto r14 = dot(row_vec.as_ref(), col_vec.as_ref());
        auto r15 = dot(row_vec, col_vec.as_ref());
        REQUIRE(r12 == 100);
        REQUIRE(r13 == 100);
        REQUIRE(r14 == 100);
        REQUIRE(r15 == 100);
        dtens u{2, 1, 2};
        dtens v{3, 3, 3};
        auto r16 = cross(u, v);
        auto r17 = cross(u.as_ref(), v);
        auto r18 = cross(u.as_ref(), v.as_ref());
        auto r19 = cross(u, v.as_ref());
        REQUIRE(r16 == dtens({-3, 0, 3}));
        REQUIRE(r17 == dtens({-3, 0, 3}));
        REQUIRE(r18 == dtens({-3, 0, 3}));
        REQUIRE(r19 == dtens({-3, 0, 3}));
        auto n1 = norm(u);
        auto n2 = norm(u.as_ref());
        REQUIRE(n1 == 3);
        REQUIRE(n2 == 3);

        auto xx = solve(AA, bb);
        REQUIRE(xx == dtens({-1, 1}));
        REQUIRE(xx.shape(0) == 2);
        auto lls_result1 = solve_lls(AA, bb);
        REQUIRE(approx_equal(lls_result1, xx));
        REQUIRE(lls_result1.shape(0) == 2);
        auto lls_result2 = bb / AA;
        REQUIRE(lls_result1 == lls_result2);
        dtens A_under{1, -1, 1, -1, 1, 1};
        A_under.reshape({2, 3});
        dtens b_under{1, 0};
        b_under.reshape({2});
        auto x_under = solve_lls(A_under, b_under);
        REQUIRE(approx_equal(x_under, dtens({1. / 4., 1. / 4., 1. / 2.})));
        REQUIRE(x_under.shape(0) == 3);
        dtens a_under_vec{1, 2};
        a_under_vec.reshape({1, 2});
        dtens b_under_scalar{2.25};
        b_under_scalar.reshape({1});
        auto x_under_vec = solve_lls(a_under_vec, b_under_scalar);
        REQUIRE(approx_equal(x_under_vec, dtens({0.45, 0.9})));
        REQUIRE(x_under_vec.shape(0) == 2);
        dtens A_over{0., 1., 0., 1.1, 0., -0.2};
        A_over.reshape({3, 2});
        dtens b_over{1.1, -1.1, -0.2};
        b_over.reshape({3});
        auto x_over = solve_lls(A_over, b_over);
        REQUIRE(approx_equal(x_over, dtens({-1.1, 1.})));
        REQUIRE(x_over.shape(0) == 2);
        dtens a_over_vec{1, 2, 8, 5};
        a_over_vec.reshape({4});
        dtens b_over_vec{3, 4, 7, 8};
        b_over_vec.reshape({4});
        dtens B_over{3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8};
        B_over.reshape({4, 4});
        auto x_over_vec = solve_lls(a_over_vec, b_over_vec);
        auto X_over_vec = solve_lls(a_over_vec, B_over);
        double over_result = 1.13829787234;
        REQUIRE(approx_equal(x_over_vec.data()[0], over_result));
        REQUIRE(x_over_vec.shape(0) == 1);
        REQUIRE(x_over_vec.size() == 1);
        dtens over_result_vec({over_result, over_result, over_result, over_result});
        over_result_vec.reshape({1, 4});
        REQUIRE(approx_equal(X_over_vec, over_result_vec));
        REQUIRE(X_over_vec.shape(0) == 1);
        REQUIRE(X_over_vec.shape(1) == 4);
        REQUIRE(X_over_vec.size() == 4);
        auto LL_inv = inv(LL);
        dtens LL_inv_result({0.6, -0.2, -0.7, 0.4});
        LL_inv_result.reshape({2, 2});
        REQUIRE(approx_equal(LL_inv, LL_inv_result));
        REQUIRE(LL_inv.shape() == LL.shape());
        auto LL_pinv = pinv(LL);
        REQUIRE(approx_equal(LL_pinv, LL_inv_result));
        REQUIRE(LL_pinv.shape() == LL.shape());
    }
    dtens CC;
    CC.reshape({4, 4});
    SECTION("row and column iterators") {
        dtens zero_row({0., 0., 0., 0.});
        zero_row.reshape({1, 4});
        for (auto &row : CC.rows()) {
            REQUIRE(approx_equal(row, zero_row));
        }
        for (auto &row : bb.rows()) {
            REQUIRE(row.size() == 1);
        }
        for (auto &row : row_vec.rows()) {
            REQUIRE(row.size() == 4);
            REQUIRE(row.shape(0) == 1);
            REQUIRE(row.shape(1) == 4);
        }
        for (auto &col : CC.cols()) {
            REQUIRE(col.size() == 4);
            REQUIRE(col.shape(0) == 4);
            REQUIRE(col.shape(1) == 1);
        }
        for (auto &col : bb.cols()) {
            REQUIRE(col.size() == 2);
            REQUIRE(col.shape(0) == 2);
            REQUIRE(col.shape(1) == 1);
        }
        for (auto &col : row_vec.cols()) {
            REQUIRE(col.size() == 1);
        }
        for (const auto &row : CC.rows()) {
            REQUIRE(approx_equal(row, zero_row));
        }
        for (const auto &row : bb.rows()) {
            REQUIRE(row.size() == 1);
        }
        for (const auto &row : row_vec.rows()) {
            REQUIRE(row.size() == 4);
            REQUIRE(row.shape(0) == 1);
            REQUIRE(row.shape(1) == 4);
        }
        for (const auto &col : CC.cols()) {
            REQUIRE(col.size() == 4);
            REQUIRE(col.shape(0) == 4);
            REQUIRE(col.shape(1) == 1);
        }
        for (const auto &col : bb.cols()) {
            REQUIRE(col.size() == 2);
            REQUIRE(col.shape(0) == 2);
            REQUIRE(col.shape(1) == 1);
        }
        for (const auto &col : row_vec.cols()) {
            REQUIRE(col.size() == 1);
        }
    }
}

TEST_CASE("Scalar Quantities", "[scalar quantities]") {
    using length_type = quantity<double, dimensions::length>;
    using time_type = quantity<double, dimensions::time>;
    using velocity_type = quantity<double, dimensions::velocity>;
    using dimensionless_type = quantity<double, dimensions::dimensionless>;
    length_type x(1);
    length_type y(5);
    time_type t(5);
    SECTION("operations on similar dimensions") {
        // add two lengths together and deduce the resulting type as length
        auto z = x + y;
        REQUIRE(z == length_type(6));
        auto w = z - x;
        REQUIRE(w == length_type(5));
        // increment length by a length
        w += x;
        REQUIRE(w == length_type(6));
        REQUIRE(std::is_same<decltype(z), length_type>::value);
        REQUIRE(std::is_same<decltype(w), length_type>::value);
        // increment dimensionless type by arithmetic type
        dimensionless_type v = 1;
        v += 1;
        REQUIRE(v == 2.);
        // increment a quantity by a scalar of a quantity
        tensor<length_type> s(1);
        z += s;
        REQUIRE(z == length_type(7));
        s += z;
        REQUIRE(s == length_type(8));
    }
    SECTION("deduce dimensions") {
        auto v = x / t; // velocity is length over time
        REQUIRE(std::is_same<decltype(v), velocity_type>::value);
        // square the quantity
        auto v_sq = pow<2>(v);
        REQUIRE(std::is_same<decltype(v_sq), decltype(v * v)>::value);
        REQUIRE(std::is_same<decltype(v_sq), quantity<double, pow_t<dimensions::velocity, 2>>>::value);
        // square root the quantity
        auto v_sq_root = root<2>(v_sq);
        REQUIRE(std::is_same<decltype(v_sq_root), decltype(v)>::value);
        // multiplication and division with literals should compile
        auto v_inv = 1 / v;
        auto v_mult = 1.0 * v;
    }
    SECTION("tensor operators with quantities") {
        tensor<quantity<double, dimensions::time>, 3, 3> A{1_s, 5_s, 6_s, 8_s, 2_s, 3_s, 3_s, 5_s, 9_s};
        tensor<quantity<double, dimensions::length>, 3> b{1_m, 2_m, 3_m};
        auto x = b / A;
        REQUIRE(std::is_same<decltype(x)::value_type, velocity_type>::value);
        x = solve(A, b);
        // get a scalar ref and initalize quantity type from it
        auto v1 = x[0];
        velocity_type v2 = v1;
        // add the scalar ref to the quantity
        v2 += v1;
        auto gg = v2 / A[0][0];
        // inverses have 1/dimension type
        auto A_inv = inv(A);
        REQUIRE(std::is_same<decltype(A_inv)::value_type,
                             quantity<double, squint::div_t<dimensions::dimensionless, dimensions::time>>>::value);
        auto x_res = inv(A) * b;
        REQUIRE(std::is_same<decltype(x_res), decltype(x)>::value);
        // add vectors of same type
        x_res += x;
        auto dotted = x_res.transpose() * x;
        REQUIRE(std::is_same<quantity<double, pow_t<dimensions::velocity, 2>>, decltype(dotted)::value_type>::value);
        auto y = x / v2;
        auto p = x * v2;
        REQUIRE(std::is_same<quantity<double, dimensions::dimensionless>, decltype(y)::value_type>::value);
        auto pp = A[0][0] - A[0][0];
        auto ppq = A[0][0] + A[0][0];
    }
    SECTION("dynamic tensor operators with quantities") {
        tensor<quantity<double, dimensions::time>, dynamic_shape> A({1_s, 5_s, 6_s, 8_s, 2_s, 3_s, 3_s, 5_s, 9_s});
        A.reshape({3, 3});
        tensor<quantity<double, dimensions::length>, dynamic_shape> b{1_m, 2_m, 3_m};
        b.reshape({3});
        auto x = b / A;
        REQUIRE(std::is_same<decltype(x)::value_type, velocity_type>::value);
        x = solve(A, b);
        // get a scalar ref and initalize quantity type from it
        auto v1 = x[0];
        velocity_type v2 = v1;
        // add the scalar ref to the quantity
        v2 += v1;
        auto gg = v2 / A[0][0];
        // inverses have 1/dimension type
        auto A_inv = inv(A);
        REQUIRE(std::is_same<decltype(A_inv)::value_type,
                             quantity<double, squint::div_t<dimensions::dimensionless, dimensions::time>>>::value);
        auto x_res = inv(A) * b;
        REQUIRE(std::is_same<decltype(x_res), decltype(x)>::value);
        // add vectors of same type
        x_res += x;
        auto dotted = x_res.transpose() * x;
        REQUIRE(std::is_same<quantity<double, pow_t<dimensions::velocity, 2>>, decltype(dotted)::value_type>::value);
        auto y = x / v2;
        auto p = x * v2;
        REQUIRE(std::is_same<quantity<double, dimensions::dimensionless>, decltype(y)::value_type>::value);
        auto pp = A[0][0] - A[0][0];
        auto ppq = A[0][0] + A[0][0];
    }
}

TEST_CASE("Dynamic tensor quantities math", "[dynamic quantity math]") {
    tensor<quantity<double, dimensions::time>, dynamic_shape> A({4, 4}, 1_s);
    tensor<quantity<double, dimensions::time>, dynamic_shape> B = A;
    tensor<quantity<double, dimensions::length>, dynamic_shape> C({1, 4}, 1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> D({4, 4}, 1_m);
    SECTION("comparisons") {
        REQUIRE(A == B);
        REQUIRE(C == D[0]);
        REQUIRE(D[0] == C);
        B[2][3] = 2_s;
        D[2][3] = 2_m;
        REQUIRE(A != B);
        REQUIRE(C != D[2]);
        REQUIRE(D[2] != C);
        B[2][3] = 1_s + 2_s * 5e-16;
        REQUIRE(approx_equal(A, B));
        REQUIRE(A != B);
        tensor<quantity<double, dimensions::length>, dynamic_shape> E({4, 4}, 1_m);
        D[2][3] = 1_m;
        E += D;
        REQUIRE(E == tensor<quantity<double, dimensions::length>, dynamic_shape>({4, 4}, 2_m));
        C += E[0];
        REQUIRE(C == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 3_m));
        E[0] += C;
        REQUIRE(E[0] == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 5_m));
        E[0] += E[0];
        REQUIRE(E[0] == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 10_m));
    }
    tensor<quantity<double, dimensions::length>, dynamic_shape> F({4, 4}, 1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> F2({4, 4}, -1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> G({4, 4}, 2_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> G0({1, 4}, 0_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> G1({1, 4}, 1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> H({1, 4}, 1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> H2({1, 4}, 2_m);
    SECTION("compound assignment operators") {
        F -= G;
        REQUIRE(F == F2);
        H -= F[0];
        REQUIRE(H == H2);
        G[0] -= H;
        REQUIRE(G[0] == G0);
        G[0] -= F[0];
        REQUIRE(G[0] == G1);
    }
    tensor<quantity<double, dimensions::length>, dynamic_shape> s5({1}, 1_m);
    tensor<quantity<double, dimensions::length>, dynamic_shape> s6({1}, 2_m);
    using dynamic_length = tensor<quantity<double, dimensions::length>, dynamic_shape>;
    using dynamic_dimensionless = tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>;
    using dynamic_area = tensor<quantity<double, pow_t<dimensions::length, 2>>, dynamic_shape>;
    dynamic_length row_vec({1, 4}, 5_m);
    dynamic_length col_vec({4}, 5_m);
    dynamic_length mat({4, 4}, 2_m);
    dynamic_length tens({4, 4, 4}, 2_m);
    G[0][0] = 0.5_m;
    SECTION("operations") {
        auto s7 = s5 + s6;
        REQUIRE(s7 == 3_m);
        auto s8 = s5 + G[0][0];
        REQUIRE(s8 == 1.5_m);
        tensor<length> s9 = s5 + 1_m;
        REQUIRE(s9 == 2_m);
        auto s10 = G[0][0] + s5;
        REQUIRE(s10 == 1.5_m);
        auto s11 = G[0][0] + G[0][0];
        REQUIRE(s11 == 1_m);
        auto s12 = G[0][0] + 1_m;
        REQUIRE(s12 == 1.5_m);
        auto s13 = 1_m + s6;
        REQUIRE(s13 == 3_m);
        auto s14 = 1_m + G[0][0];
        REQUIRE(s14 == 1.5_m);
        s7 = s5 - s6;
        REQUIRE(s7 == -1_m);
        s8 = s5 - G[0][0];
        REQUIRE(s8 == 0.5_m);
        s9 = s5 - 1_m;
        REQUIRE(s9 == 0_m);
        s10 = G[0][0] - s5;
        REQUIRE(s10 == -0.5_m);
        s11 = G[0][0] - G[0][0];
        REQUIRE(s11 == 0_m);
        s12 = G[0][0] - 1_m;
        REQUIRE(s12 == -0.5_m);
        s13 = 1_m - s6;
        REQUIRE(s13 == -1_m);
        s14 = 1_m - G[0][0];
        REQUIRE(s14 == 0.5_m);
        auto area = s5 * s6;
        auto area2 = dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(2));
        REQUIRE(area == area2);
        auto ppp = s5 * G[0][0];
        auto d8 = s5 * G[0][0];
        REQUIRE(d8 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d9 = s5 * 1_m;
        REQUIRE(d9 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(1)));
        auto d10 = G[0][0] * s5;
        REQUIRE(d10 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d11 = G[0][0] * G[0][0];
        REQUIRE(d11 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(0.25)));
        auto d12 = G[0][0] * 1_m;
        REQUIRE(d12 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d13 = 1_m * s6;
        REQUIRE(d13 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(2)));
        auto d14 = 1_m * G[0][0];
        REQUIRE(d14 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d7 = s5 / s6;
        REQUIRE(d7 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 0.5));
        auto p8 = s5 / G[0][0];
        REQUIRE(p8 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 2));
        auto p9 = s5 / 1;
        REQUIRE(p9 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1}, 1_m));
        auto p10 = G[0][0] / s5;
        REQUIRE(p10 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 0.5));
        auto p11 = G[0][0] / G[0][0];
        REQUIRE(p11 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 1));
        auto p12 = G[0][0] / 1;
        REQUIRE(p12 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1}, 0.5_m));
        auto p13 = tensor<quantity<double, dimensions::length>, dynamic_shape>({1}, 1_m) / s6;
        REQUIRE(p13 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 0.5));
        auto p14 = tensor<quantity<double, dimensions::length>, dynamic_shape>({1}, 1_m) / G[0][0];
        REQUIRE(p14 == tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 2));
        tensor<quantity<double, dimensions::length>, dynamic_shape> t1({1, 4}, 1_m);
        tensor<quantity<double, dimensions::length>, dynamic_shape> t2({1, 4}, 2_m);
        tensor<quantity<double, dimensions::length>, dynamic_shape> t3({4, 4}, 2_m);
        auto t4 = t1 + t2;
        auto t5 = t1 + t3[0];
        auto t6 = t3[0] + t1;
        auto t7 = t3[0] + t3[0];
        REQUIRE(t4 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 3_m));
        REQUIRE(t5 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 3_m));
        REQUIRE(t6 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 3_m));
        REQUIRE(t7 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 4_m));
        t4 = t1 - t2;
        t5 = t1 - t3[0];
        t6 = t3[0] - t1;
        t7 = t3[0] - t3[0];
        REQUIRE(t4 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, -1_m));
        REQUIRE(t5 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, -1_m));
        REQUIRE(t6 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 1_m));
        REQUIRE(t7 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, 0_m));
        auto s15 = -s14;
        auto s16 = -t3[0][0];
        auto t8 = -t3[0];
        auto t9 = -t3;
        REQUIRE(s15 == length(-0.5));
        REQUIRE(s16 == length(-2));
        REQUIRE(t8 == tensor<quantity<double, dimensions::length>, dynamic_shape>({1, 4}, -2_m));
        REQUIRE(t9 == tensor<quantity<double, dimensions::length>, dynamic_shape>({4, 4}, -2_m));
        auto r1 = row_vec * mat;
        auto r1r = row_vec * tens[0].simplify_shape();
        auto r1rr = col_vec.transpose() * tens[0].simplify_shape();
        auto r1rrr = col_vec.transpose() * mat;
        REQUIRE(r1 == dynamic_area({1, 4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1r == dynamic_area({1, 4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1rr == dynamic_area({1, 4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1rrr == dynamic_area({1, 4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        auto r2 = mat * col_vec;
        auto r2r = mat * row_vec.transpose();
        auto r2rr = mat.as_shape({4, 4}) * col_vec;
        auto r2rrr = mat.as_shape({4, 4}) * col_vec.as_shape({4});
        REQUIRE(r2 == dynamic_area({4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2r == dynamic_area({4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2rr == dynamic_area({4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2rrr == dynamic_area({4}, quantity<double, pow_t<dimensions::length, 2>>(40)));
        auto r3 = row_vec * col_vec;
        auto r3r = row_vec.as_ref() * col_vec;
        auto r3rr = row_vec.as_ref() * col_vec.as_ref();
        auto r3rrr = row_vec * col_vec.as_ref();
        REQUIRE(r3 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3r == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3rr == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3rrr == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        auto r4 = col_vec * row_vec;
        auto r4r = col_vec.as_ref() * row_vec;
        auto r4rr = col_vec.as_ref() * row_vec.as_ref();
        auto r4rrr = col_vec * row_vec.as_ref();
        REQUIRE(r4 == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4r == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4rr == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4rrr == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(25)));
        auto r5 = mat * mat;
        auto r5r = mat.transpose() * mat;
        auto r5rr = mat.transpose() * mat.transpose();
        auto r5rrr = mat * mat.transpose();
        REQUIRE(r5 == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5r == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5rr == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5rrr == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(16)));
        auto r6 = mat * tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 2);
        REQUIRE(r6 == dynamic_length({4, 4}, quantity<double, dimensions::length>(4)));
        auto r8 = mat * mat[0][0];
        REQUIRE(r8 == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(4)));
        auto r9 = mat.as_ref() * tensor<quantity<double, dimensions::dimensionless>, dynamic_shape>({1}, 2);
        REQUIRE(r9 == dynamic_length({4, 4}, quantity<double, dimensions::length>(4)));
        auto r11 = mat.as_ref() * mat[0][0];
        REQUIRE(r11 == dynamic_area({4, 4}, quantity<double, pow_t<dimensions::length, 2>>(4)));
    }
    dynamic_length AA{4_m, 2_m, 7_m, 6_m};
    AA.reshape({2, 2});
    auto LL = AA;
    dynamic_length bb{3_m, 4_m};
    SECTION("functions") {
        auto r12 = dot(row_vec, col_vec);
        auto r13 = dot(row_vec, col_vec.as_ref());
        auto r14 = dot(row_vec.as_ref(), col_vec.as_ref());
        auto r15 = dot(row_vec, col_vec.as_ref());
        REQUIRE(r12 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r13 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r14 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r15 == dynamic_area({1}, quantity<double, pow_t<dimensions::length, 2>>(100)));
        tensor<quantity<double, dimensions::length>, dynamic_shape> u{2_m, 1_m, 2_m};
        tensor<quantity<double, dimensions::dimensionless>, dynamic_shape> v{3, 3, 3};
        auto r16 = cross(u, v);
        auto r17 = cross(u.as_ref(), v);
        auto r18 = cross(u.as_ref(), v.as_ref());
        auto r19 = cross(u, v.as_ref());
        REQUIRE(r16 == dynamic_length({-3_m, 0_m, 3_m}));
        REQUIRE(r17 == dynamic_length({-3_m, 0_m, 3_m}));
        REQUIRE(r18 == dynamic_length({-3_m, 0_m, 3_m}));
        REQUIRE(r19 == dynamic_length({-3_m, 0_m, 3_m}));
        auto n1 = norm(u);
        auto n2 = norm(u.as_ref());
        REQUIRE(n1 == 3_m);
        REQUIRE(n2 == 3_m);
        auto xx = solve(AA, bb);
        REQUIRE(xx == dynamic_dimensionless({-1, 1}));
        REQUIRE(xx.shape(0) == 2);
        auto lls_result1 = solve_lls(AA, bb);
        REQUIRE(approx_equal(lls_result1, xx));
        REQUIRE(lls_result1.shape(0) == 2);
        auto lls_result2 = bb / AA;
        REQUIRE(lls_result1 == lls_result2);
        dynamic_dimensionless A_under{1, -1, 1, -1, 1, 1};
        A_under.reshape({2, 3});
        dynamic_dimensionless b_under{1, 0};
        b_under.reshape({2});
        auto x_under = solve_lls(A_under, b_under);
        REQUIRE(approx_equal(x_under, dynamic_dimensionless({1. / 4., 1. / 4., 1. / 2.})));
        REQUIRE(x_under.shape(0) == 3);
        dynamic_dimensionless a_under_vec{1, 2};
        a_under_vec.reshape({1, 2});
        dynamic_dimensionless b_under_scalar{2.25};
        b_under_scalar.reshape({1});
        auto x_under_vec = solve_lls(a_under_vec, b_under_scalar);
        REQUIRE(approx_equal(x_under_vec, dynamic_dimensionless({0.45, 0.9})));
        REQUIRE(x_under_vec.shape(0) == 2);
        dynamic_dimensionless A_over{0., 1., 0., 1.1, 0., -0.2};
        A_over.reshape({3, 2});
        dynamic_dimensionless b_over{1.1, -1.1, -0.2};
        b_over.reshape({3});
        auto x_over = solve_lls(A_over, b_over);
        REQUIRE(approx_equal(x_over, dynamic_dimensionless({-1.1, 1.})));
        REQUIRE(x_over.shape(0) == 2);
        dynamic_dimensionless a_over_vec{1, 2, 8, 5};
        a_over_vec.reshape({4});
        dynamic_dimensionless b_over_vec{3, 4, 7, 8};
        b_over_vec.reshape({4});
        dynamic_dimensionless B_over{3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8};
        B_over.reshape({4, 4});
        auto x_over_vec = solve_lls(a_over_vec, b_over_vec);
        auto X_over_vec = solve_lls(a_over_vec, B_over);
        quantity<double, dimensions::dimensionless> over_result = 1.13829787234;
        REQUIRE(approx_equal(x_over_vec.data()[0], over_result));
        REQUIRE(x_over_vec.shape(0) == 1);
        REQUIRE(x_over_vec.size() == 1);
        dynamic_dimensionless over_result_vec({over_result, over_result, over_result, over_result});
        over_result_vec.reshape({1, 4});
        REQUIRE(approx_equal(X_over_vec, over_result_vec));
        REQUIRE(X_over_vec.shape(0) == 1);
        REQUIRE(X_over_vec.shape(1) == 4);
        REQUIRE(X_over_vec.size() == 4);
        auto LL_inv = inv(LL);
        dynamic_dimensionless LL_inv_result({0.6, -0.2, -0.7, 0.4});
        LL_inv_result.reshape({2, 2});
        REQUIRE(approx_equal(LL_inv, LL_inv_result));
        REQUIRE(LL_inv.shape() == LL.shape());
        auto LL_pinv = pinv(LL);
        REQUIRE(approx_equal(LL_pinv, LL_inv_result));
        REQUIRE(LL_pinv.shape() == LL.shape());
    }
}
template <int... sizes> using fixed_length = tensor<quantity<double, dimensions::length>, sizes...>;
template <int... sizes> using fixed_dimensionless = tensor<quantity<double, dimensions::dimensionless>, sizes...>;
template <int... sizes> using fixed_area = tensor<quantity<double, pow_t<dimensions::length, 2>>, sizes...>;
TEST_CASE("Fixed tensor quantities math", "[fixed quantity math]") {
    tensor<quantity<double, dimensions::time>, 4, 4> A(1_s);
    tensor<quantity<double, dimensions::time>, 4, 4> B = A;
    tensor<quantity<double, dimensions::length>, 1, 4> C(1_m);
    tensor<quantity<double, dimensions::length>, 4, 4> D(1_m);
    SECTION("comparisons") {
        REQUIRE(A == B);
        REQUIRE(C == D[0]);
        REQUIRE(D[0] == C);
        B[2][3] = 2_s;
        D[2][3] = 2_m;
        REQUIRE(A != B);
        REQUIRE(C != D[2]);
        REQUIRE(D[2] != C);
        B[2][3] = 1_s + 2_s * 5e-16;
        REQUIRE(approx_equal(A, B));
        REQUIRE(A != B);
        tensor<quantity<double, dimensions::length>, 4, 4> E(1_m);
        D[2][3] = 1_m;
        E += D;
        REQUIRE(E == tensor<quantity<double, dimensions::length>, 4, 4>(2_m));
        C += E[0];
        REQUIRE(C == tensor<quantity<double, dimensions::length>, 1, 4>(3_m));
        E[0] += C;
        REQUIRE(E[0] == tensor<quantity<double, dimensions::length>, 1, 4>(5_m));
        E[0] += E[0];
        REQUIRE(E[0] == tensor<quantity<double, dimensions::length>, 1, 4>(10_m));
    }
    tensor<quantity<double, dimensions::length>, 4, 4> F(1_m);
    tensor<quantity<double, dimensions::length>, 4, 4> F2(-1_m);
    tensor<quantity<double, dimensions::length>, 4, 4> G(2_m);
    tensor<quantity<double, dimensions::length>, 1, 4> G0(0_m);
    tensor<quantity<double, dimensions::length>, 1, 4> G1(1_m);
    tensor<quantity<double, dimensions::length>, 1, 4> H(1_m);
    tensor<quantity<double, dimensions::length>, 1, 4> H2(2_m);
    SECTION("compound assignment operators") {
        F -= G;
        REQUIRE(F == F2);
        H -= F[0];
        REQUIRE(H == H2);
        G[0] -= H;
        REQUIRE(G[0] == G0);
        G[0] -= F[0];
        REQUIRE(G[0] == G1);
    }
    tensor<quantity<double, dimensions::length>> s5(1_m);
    tensor<quantity<double, dimensions::length>> s6(2_m);
    fixed_length<1, 4> row_vec(5_m);
    fixed_length<4> col_vec(5_m);
    fixed_length<4, 4> mat(2_m);
    fixed_length<4, 4, 4> tens(2_m);
    G[0][0] = 0.5_m;
    SECTION("operations") {
        auto s7 = s5 + s6;
        REQUIRE(s7 == 3_m);
        auto s8 = s5 + G[0][0];
        REQUIRE(s8 == 1.5_m);
        tensor<length> s9 = s5 + 1_m;
        REQUIRE(s9 == 2_m);
        auto s10 = G[0][0] + s5;
        REQUIRE(s10 == 1.5_m);
        auto s11 = G[0][0] + G[0][0];
        REQUIRE(s11 == 1_m);
        auto s12 = G[0][0] + 1_m;
        REQUIRE(s12 == 1.5_m);
        auto s13 = 1_m + s6;
        REQUIRE(s13 == 3_m);
        auto s14 = 1_m + G[0][0];
        REQUIRE(s14 == 1.5_m);
        s7 = s5 - s6;
        REQUIRE(s7 == -1_m);
        s8 = s5 - G[0][0];
        REQUIRE(s8 == 0.5_m);
        s9 = s5 - 1_m;
        REQUIRE(s9 == 0_m);
        s10 = G[0][0] - s5;
        REQUIRE(s10 == -0.5_m);
        s11 = G[0][0] - G[0][0];
        REQUIRE(s11 == 0_m);
        s12 = G[0][0] - 1_m;
        REQUIRE(s12 == -0.5_m);
        s13 = 1_m - s6;
        REQUIRE(s13 == -1_m);
        s14 = 1_m - G[0][0];
        REQUIRE(s14 == 0.5_m);
        auto area = s5 * s6;
        auto area2 = fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(2));
        REQUIRE(area == area2);
        auto ppp = s5 * G[0][0];
        auto d8 = s5 * G[0][0];
        REQUIRE(d8 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d9 = s5 * 1_m;
        REQUIRE(d9 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(1)));
        auto d10 = G[0][0] * s5;
        REQUIRE(d10 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d11 = G[0][0] * G[0][0];
        REQUIRE(d11 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(0.25)));
        auto d12 = G[0][0] * 1_m;
        REQUIRE(d12 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d13 = 1_m * s6;
        REQUIRE(d13 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(2)));
        auto d14 = 1_m * G[0][0];
        REQUIRE(d14 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(0.5)));
        auto d7 = s5 / s6;
        REQUIRE(d7 == tensor<quantity<double, dimensions::dimensionless>>(0.5));
        auto p8 = s5 / G[0][0];
        REQUIRE(p8 == tensor<quantity<double, dimensions::dimensionless>>(2));
        auto p9 = s5 / 1;
        REQUIRE(p9 == tensor<quantity<double, dimensions::length>>(1_m));
        auto p10 = G[0][0] / s5;
        REQUIRE(p10 == tensor<quantity<double, dimensions::dimensionless>>(0.5));
        auto p11 = G[0][0] / G[0][0];
        REQUIRE(p11 == tensor<quantity<double, dimensions::dimensionless>>(1));
        auto p12 = G[0][0] / 1;
        REQUIRE(p12 == tensor<quantity<double, dimensions::length>>(0.5_m));
        auto p13 = tensor<quantity<double, dimensions::length>>(1_m) / s6;
        REQUIRE(p13 == tensor<quantity<double, dimensions::dimensionless>>(0.5));
        auto p14 = tensor<quantity<double, dimensions::length>>(1_m) / G[0][0];
        REQUIRE(p14 == tensor<quantity<double, dimensions::dimensionless>>(2));
        tensor<quantity<double, dimensions::length>, 1, 4> t1(1_m);
        tensor<quantity<double, dimensions::length>, 1, 4> t2(2_m);
        tensor<quantity<double, dimensions::length>, 4, 4> t3(2_m);
        auto t4 = t1 + t2;
        auto t5 = t1 + t3[0];
        auto t6 = t3[0] + t1;
        auto t7 = t3[0] + t3[0];
        REQUIRE(t4 == tensor<quantity<double, dimensions::length>, 1, 4>(3_m));
        REQUIRE(t5 == tensor<quantity<double, dimensions::length>, 1, 4>(3_m));
        REQUIRE(t6 == tensor<quantity<double, dimensions::length>, 1, 4>(3_m));
        REQUIRE(t7 == tensor<quantity<double, dimensions::length>, 1, 4>(4_m));
        t4 = t1 - t2;
        t5 = t1 - t3[0];
        t6 = t3[0] - t1;
        t7 = t3[0] - t3[0];
        REQUIRE(t4 == tensor<quantity<double, dimensions::length>, 1, 4>(-1_m));
        REQUIRE(t5 == tensor<quantity<double, dimensions::length>, 1, 4>(-1_m));
        REQUIRE(t6 == tensor<quantity<double, dimensions::length>, 1, 4>(1_m));
        REQUIRE(t7 == tensor<quantity<double, dimensions::length>, 1, 4>(0_m));
        auto s15 = -s14;
        auto s16 = -t3[0][0];
        auto t8 = -t3[0];
        auto t9 = -t3;
        REQUIRE(s15 == length(-0.5));
        REQUIRE(s16 == length(-2));
        REQUIRE(t8 == tensor<quantity<double, dimensions::length>, 1, 4>(-2_m));
        REQUIRE(t9 == tensor<quantity<double, dimensions::length>, 4, 4>(-2_m));
        auto r1 = row_vec * mat;
        auto r1r = row_vec * tens[0].simplify_shape();
        auto r1rr = col_vec.transpose() * tens[0].simplify_shape();
        auto r1rrr = col_vec.transpose() * mat;
        REQUIRE(r1 == fixed_area<1, 4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1r == fixed_area<1, 4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1rr == fixed_area<1, 4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r1rrr == fixed_area<1, 4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        auto r2 = mat * col_vec;
        auto r2r = mat * row_vec.transpose();
        auto r2rr = mat.as_shape<4, 4>() * col_vec;
        auto r2rrr = mat.as_shape<4, 4>() * col_vec.as_shape<4>();
        REQUIRE(r2 == fixed_area<4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2r == fixed_area<4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2rr == fixed_area<4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        REQUIRE(r2rrr == fixed_area<4>(quantity<double, pow_t<dimensions::length, 2>>(40)));
        auto r3 = row_vec * col_vec;
        auto r3r = row_vec.as_ref() * col_vec;
        auto r3rr = row_vec.as_ref() * col_vec.as_ref();
        auto r3rrr = row_vec * col_vec.as_ref();
        REQUIRE(r3 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3r == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3rr == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r3rrr == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        auto r4 = col_vec * row_vec;
        auto r4r = col_vec.as_ref() * row_vec;
        auto r4rr = col_vec.as_ref() * row_vec.as_ref();
        auto r4rrr = col_vec * row_vec.as_ref();
        REQUIRE(r4 == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4r == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4rr == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(25)));
        REQUIRE(r4rrr == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(25)));
        auto r5 = mat * mat;
        auto r5r = mat.transpose() * mat;
        auto r5rr = mat.transpose() * mat.transpose();
        auto r5rrr = mat * mat.transpose();
        REQUIRE(r5 == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5r == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5rr == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(16)));
        REQUIRE(r5rrr == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(16)));
        auto r6 = mat * tensor<quantity<double, dimensions::dimensionless>>(2);
        REQUIRE(r6 == fixed_length<4, 4>(quantity<double, dimensions::length>(4)));
        auto r8 = mat * mat[0][0];
        REQUIRE(r8 == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(4)));
        auto r9 = mat.as_ref() * tensor<quantity<double, dimensions::dimensionless>>(2);
        REQUIRE(r9 == fixed_length<4, 4>(quantity<double, dimensions::length>(4)));
        auto r11 = mat.as_ref() * mat[0][0];
        REQUIRE(r11 == fixed_area<4, 4>(quantity<double, pow_t<dimensions::length, 2>>(4)));
    }
    fixed_length<2, 2> AA{4_m, 2_m, 7_m, 6_m};
    auto LL = AA;
    fixed_length<2> bb{3_m, 4_m};
    SECTION("functions") {
        auto r12 = dot(row_vec, col_vec);
        auto r13 = dot(row_vec, col_vec.as_ref());
        auto r14 = dot(row_vec.as_ref(), col_vec.as_ref());
        auto r15 = dot(row_vec, col_vec.as_ref());
        REQUIRE(r12 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r13 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r14 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        REQUIRE(r15 == fixed_area<>(quantity<double, pow_t<dimensions::length, 2>>(100)));
        tensor<quantity<double, dimensions::length>, 3> u{2_m, 1_m, 2_m};
        tensor<quantity<double, dimensions::dimensionless>, 3> v{3, 3, 3};
        auto r16 = cross(u, v);
        auto r17 = cross(u.as_ref(), v);
        auto r18 = cross(u.as_ref(), v.as_ref());
        auto r19 = cross(u, v.as_ref());
        REQUIRE(r16 == fixed_length<3>({-3_m, 0_m, 3_m}));
        REQUIRE(r17 == fixed_length<3>({-3_m, 0_m, 3_m}));
        REQUIRE(r18 == fixed_length<3>({-3_m, 0_m, 3_m}));
        REQUIRE(r19 == fixed_length<3>({-3_m, 0_m, 3_m}));
        auto n1 = norm(u);
        auto n2 = norm(u.as_ref());
        REQUIRE(n1 == 3_m);
        REQUIRE(n2 == 3_m);
        auto xx = solve(AA, bb);
        REQUIRE(xx == fixed_dimensionless<2>({-1, 1}));
        REQUIRE(xx.shape(0) == 2);
        auto lls_result1 = solve_lls(AA, bb);
        REQUIRE(approx_equal(lls_result1, xx));
        REQUIRE(lls_result1.shape(0) == 2);
        auto lls_result2 = bb / AA;
        REQUIRE(lls_result1 == lls_result2);
        fixed_dimensionless<2, 3> A_under{1, -1, 1, -1, 1, 1};
        fixed_dimensionless<2> b_under{1, 0};
        auto x_under = solve_lls(A_under, b_under);
        REQUIRE(approx_equal(x_under, fixed_dimensionless<3>({1. / 4., 1. / 4., 1. / 2.})));
        REQUIRE(x_under.shape(0) == 3);
        fixed_dimensionless<1, 2> a_under_vec{1, 2};
        fixed_dimensionless<> b_under_scalar{2.25};
        auto x_under_vec = solve_lls(a_under_vec, b_under_scalar);
        REQUIRE(approx_equal(x_under_vec, fixed_dimensionless<2>({0.45, 0.9})));
        REQUIRE(x_under_vec.shape(0) == 2);
        fixed_dimensionless<3, 2> A_over{0., 1., 0., 1.1, 0., -0.2};
        fixed_dimensionless<3> b_over{1.1, -1.1, -0.2};
        auto x_over = solve_lls(A_over, b_over);
        REQUIRE(approx_equal(x_over, fixed_dimensionless<2>({-1.1, 1.})));
        REQUIRE(x_over.shape(0) == 2);
        fixed_dimensionless<4> a_over_vec{1, 2, 8, 5};
        fixed_dimensionless<4> b_over_vec{3, 4, 7, 8};
        fixed_dimensionless<4, 4> B_over{3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8, 3, 4, 7, 8};
        auto x_over_vec = solve_lls(a_over_vec, b_over_vec);
        auto X_over_vec = solve_lls(a_over_vec, B_over);
        quantity<double, dimensions::dimensionless> over_result = 1.13829787234;
        REQUIRE(approx_equal(x_over_vec.data()[0], over_result));
        REQUIRE(x_over_vec.shape(0) == 1);
        REQUIRE(x_over_vec.size() == 1);
        fixed_dimensionless<1, 4> over_result_vec({over_result, over_result, over_result, over_result});
        REQUIRE(approx_equal(X_over_vec, over_result_vec));
        REQUIRE(X_over_vec.shape(0) == 1);
        REQUIRE(X_over_vec.shape(1) == 4);
        REQUIRE(X_over_vec.size() == 4);
        auto LL_inv = inv(LL);
        fixed_dimensionless<2, 2> LL_inv_result({0.6, -0.2, -0.7, 0.4});
        REQUIRE(approx_equal(LL_inv, LL_inv_result));
        REQUIRE(LL_inv.shape() == LL.shape());
        auto LL_pinv = pinv(LL);
        REQUIRE(approx_equal(LL_pinv, LL_inv_result));
        REQUIRE(LL_pinv.shape() == LL.shape());
        // invert matricies that need pivoting
        auto view = look_at(dvec3{0.0, 0.0, 5.0}, dvec3{0., 0., 0.}, dvec3{1., 0., 0.});
        auto view_inv = inv(view);
        auto view_lls_inv = dmat4::I() / view;
        REQUIRE(approx_equal(view_inv, view_lls_inv));
        dmat4 view_inv_result{dvec4{0.0, -1.0, 0.0, 0.0}, dvec4{1.0, 0.0, 0.0, 0.0}, dvec4{0.0, 0.0, 1.0, 0.0},
                              dvec4{0.0, 0.0, 5.0, 1.0}};
        REQUIRE(approx_equal(view_inv, view_inv_result));
    }
}

TEST_CASE("Transformation Matrices", "[transformation matrices]") {
    // Translate
    dmat4 T1 = dmat4::I();
    dvec3 x{1, 2, 3};
    auto T2 = translate(T1, x);
    dmat4 T2_result{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1};
    REQUIRE(T2 == T2_result);
    auto T3 = translate(T2, x);
    dmat4 T3_result{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 4, 6, 1};
    REQUIRE(T3 == T3_result);
    // Scale
    auto T4 = scale(T3, x);
    dmat4 T4_result{1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 2, 4, 6, 1};
    REQUIRE(T4 == T4_result);
    // look at
    auto T5 = look_at(dvec3(0.), dvec3{0., 1., 0.}, dvec3{0., 0., 1.});
    dmat4 T5_result{drvec4{1., 0., 0., 0.}, drvec4{0., 0., 1., 0.}, drvec4{0., -1., 0., 0.}, drvec4{0., 0., 0., 1.}};
    REQUIRE(approx_equal(T5, T5_result));
    auto T7 = look_at(fvec3{3.0F, 0.5F, 0.0F}, fvec3{0.F, 0.F, 0.F}, fvec3{0.F, 0.F, 1.F});
    // check for numerical stability after double inversion
    auto T7inv = inv(T7);
    auto T7invinv = inv(T7inv);
    REQUIRE(approx_equal(T7invinv, T7));
    // rotate
    auto T6 = rotate(T5, M_PI_2, dvec3{0.F, 0.F, 1.F});
    dmat4 T6_result{dvec4{0., 1., 0., 0.}, dvec4{0., 0., -1., 0.}, dvec4{-1., 0., 0., 0.}, dvec4{0., 0., 0., 1.}};
    REQUIRE(approx_equal(T6, T6_result));
}

TEST_CASE("Projection Matrices", "[projection matrices]") {
    // Orthographic projection
    dmat4 ortho_proj = ortho(-1., 1., -1., 1., -1., 1.);
    dmat4 ortho_proj_result;
    ortho_proj_result[0][0] = 1.;
    ortho_proj_result[1][1] = 1.;
    ortho_proj_result[2][2] = -1.;
    ortho_proj_result[3][3] = 1.;
    REQUIRE(approx_equal(ortho_proj, ortho_proj_result));
    // Perspective projection
    dmat4 persp_proj = perspective(M_PI_4, 1., 0., 1.);
    dmat4 persp_proj_result;
    double tan_half_fovy = std::tan(M_PI_4 / 2.);
    persp_proj_result[0][0] = 1. / tan_half_fovy;
    persp_proj_result[1][1] = persp_proj_result[0][0];
    persp_proj_result[2][2] = -1.;
    persp_proj_result[3][2] = -1.;
    persp_proj_result[2][3] = 0.;
    REQUIRE(approx_equal(persp_proj, persp_proj_result));
}

area norm_squared(const tensor<length, 3> &x) { return pow<2>(norm(x)); }
auto norm_squared_grad(const tensor<length, 3> &x) { return grad(norm_squared, x); }

TEST_CASE("opimization", "[optimization]") {
    tensor<length, 3> x{1_m, 2_m, 3_m};
    auto g = grad(norm_squared, x);
    jac(norm_squared_grad, x);
}

TEST_CASE("fixed tensors static methods", "[fixed static]") {
    auto ones = dmat4::ones();
    auto zeros = dmat4::zeros();
    auto eye = dmat4::I();
    auto twos = dmat4::fill(2.);
    auto diag_vec = dmat4::diag({1., 2., 3., 4.});
    auto diag_scalar = dmat4::diag(2.);
    for (const auto &v : ones) {
        REQUIRE(v == 1.);
    }
    for (const auto &v : zeros) {
        REQUIRE(v == 0.);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(eye[i][i] == 1.);
    }
    for (const auto &v : twos) {
        REQUIRE(v == 2.);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(diag_vec[i][i] == i + 1);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(diag_scalar[i][i] == 2.);
    }
    auto rand = dmat4::random(5, 6);
    for (const auto &v : rand) {
        REQUIRE(v >= 5.);
        REQUIRE(v <= 6.);
    }
}

TEST_CASE("dynamic tensors static methods", "[dynamic static]") {
    auto ones = tensor<double, dynamic_shape>::ones({4, 4});
    auto zeros = tensor<double, dynamic_shape>::zeros({4, 4});
    auto eye = tensor<double, dynamic_shape>::I(4);
    auto twos = tensor<double, dynamic_shape>::fill({4, 4}, 2.);
    auto diag_vec = tensor<double, dynamic_shape>::diag({1., 2., 3., 4.});
    auto diag_scalar = tensor<double, dynamic_shape>::diag(4, 2.);
    for (const auto &v : ones) {
        REQUIRE(v == 1.);
    }
    for (const auto &v : zeros) {
        REQUIRE(v == 0.);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(eye[i][i] == 1.);
    }
    for (const auto &v : twos) {
        REQUIRE(v == 2.);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(diag_vec[i][i] == i + 1);
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(diag_scalar[i][i] == 2.);
    }
    auto rand = tensor<double, dynamic_shape>::random({4, 4}, 5, 6);
    for (const auto &v : rand) {
        REQUIRE(v >= 5.);
        REQUIRE(v <= 6.);
    }
}