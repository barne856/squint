// NOLINTBEGIN
#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"

using namespace squint;

TEST_CASE("Element-wise operations") {
    SUBCASE("Fixed shape tensors") {
        tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
        tensor<float, shape<2, 3>> b({2, 5, 3, 6, 4, 7});

        SUBCASE("Addition") {
            auto c = a + b;
            CHECK(c == tensor<float, shape<2, 3>>({3, 9, 5, 11, 7, 13}));
        }

        SUBCASE("Subtraction") {
            auto c = a - b;
            CHECK(c == tensor<float, shape<2, 3>>({-1, -1, -1, -1, -1, -1}));
        }

        SUBCASE("Equality and Inequality") {
            CHECK(a == a);
            CHECK(a != b);
        }

        SUBCASE("Unary negation") {
            auto c = -a;
            CHECK(c == tensor<float, shape<2, 3>>({-1, -4, -2, -5, -3, -6}));
        }

        SUBCASE("In-place addition") {
            a += b;
            CHECK(a == tensor<float, shape<2, 3>>({3, 9, 5, 11, 7, 13}));
        }

        SUBCASE("In-place subtraction") {
            a -= b;
            CHECK(a == tensor<float, shape<2, 3>>({-1, -1, -1, -1, -1, -1}));
        }
    }

    SUBCASE("Dynamic shape tensors") {
        tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        tensor<float, dynamic, dynamic> b({2, 3}, std::vector<float>{2, 5, 3, 6, 4, 7});

        SUBCASE("Addition") {
            auto c = a + b;
            CHECK(c == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{3, 9, 5, 11, 7, 13}));
        }

        SUBCASE("Subtraction") {
            auto c = a - b;
            CHECK(c == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{-1, -1, -1, -1, -1, -1}));
        }

        SUBCASE("Equality and Inequality") {
            CHECK(a == a);
            CHECK(a != b);
        }

        SUBCASE("Unary negation") {
            auto c = -a;
            CHECK(c == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{-1, -4, -2, -5, -3, -6}));
        }

        SUBCASE("In-place addition") {
            a += b;
            CHECK(a == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{3, 9, 5, 11, 7, 13}));
        }

        SUBCASE("In-place subtraction") {
            a -= b;
            CHECK(a == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{-1, -1, -1, -1, -1, -1}));
        }
    }
}

TEST_CASE("Scalar operations") {
    SUBCASE("Fixed shape tensors") {
        tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});

        SUBCASE("Scalar multiplication") {
            auto b = a * 2.0f;
            CHECK(b == tensor<float, shape<2, 3>>({2, 8, 4, 10, 6, 12}));

            auto c = 2.0f * a;
            CHECK(c == b);
        }

        SUBCASE("Scalar division") {
            auto b = a / 2.0f;
            CHECK(b == tensor<float, shape<2, 3>>({0.5, 2, 1, 2.5, 1.5, 3}));
        }

        SUBCASE("In-place scalar multiplication") {
            a *= 2.0f;
            CHECK(a == tensor<float, shape<2, 3>>({2, 8, 4, 10, 6, 12}));
        }

        SUBCASE("In-place scalar division") {
            a /= 2.0f;
            CHECK(a == tensor<float, shape<2, 3>>({0.5, 2, 1, 2.5, 1.5, 3}));
        }
    }

    SUBCASE("Dynamic shape tensors") {
        tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});

        SUBCASE("Scalar multiplication") {
            auto b = a * 2.0f;
            CHECK(b == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{2, 8, 4, 10, 6, 12}));

            auto c = 2.0f * a;
            CHECK(c == b);
        }

        SUBCASE("Scalar division") {
            auto b = a / 2.0f;
            CHECK(b == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{0.5, 2, 1, 2.5, 1.5, 3}));
        }

        SUBCASE("In-place scalar multiplication") {
            a *= 2.0f;
            CHECK(a == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{2, 8, 4, 10, 6, 12}));
        }

        SUBCASE("In-place scalar division") {
            a /= 2.0f;
            CHECK(a == tensor<float, dynamic, dynamic>({2, 3}, std::vector<float>{0.5, 2, 1, 2.5, 1.5, 3}));
        }
    }
}

TEST_CASE("Matrix multiplication") {
    SUBCASE("Fixed shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto c = a.transpose() * b;
            CHECK(c == tensor<float, shape<1, 1>>({32}));
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto c = a * b.transpose();
            CHECK(c == tensor<float, shape<3, 3>>({4, 8, 12, 5, 10, 15, 6, 12, 18}));
        }

        SUBCASE("Vector times matrix") {
            tensor<float, shape<2>> a({1, 2});
            tensor<float, shape<2, 3>> b({1, 4, 2, 5, 3, 6});
            auto c = a.transpose() * b;
            CHECK(c == tensor<float, shape<1, 3>>({9, 12, 15}));
        }

        SUBCASE("Matrix times vector") {
            tensor<float, shape<3, 3>> a({1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, shape<3>> b({1, 2, 3});
            auto c = a * b;
            CHECK(c == tensor<float, shape<3>>({30, 36, 42}));
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
            tensor<float, shape<3, 2>> b({1, 4, 2, 5, 3, 6});
            auto c = a * b;
            CHECK(c == tensor<float, shape<2, 2>>({15, 36, 29, 71}));
        }
    }

    SUBCASE("Dynamic shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto c = a.transpose() * b;
            CHECK(c.size() == 1);
            CHECK(c(0, 0) == 32);
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto c = a * b.transpose();
            CHECK(c == tensor<float, dynamic, dynamic>({3, 3}, std::vector<float>{4, 8, 12, 5, 10, 15, 6, 12, 18}));
        }

        SUBCASE("Vector times matrix") {
            tensor<float, dynamic, dynamic> a({2}, std::vector<float>{1, 2});
            tensor<float, dynamic, dynamic> b({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto c = a.transpose() * b;
            CHECK(c == tensor<float, dynamic, dynamic>({1, 3}, std::vector<float>{9, 12, 15}));
        }

        SUBCASE("Matrix times vector") {
            tensor<float, dynamic, dynamic> a({3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{1, 2, 3});
            auto c = a * b;
            CHECK(c == tensor<float, dynamic, dynamic>({2}, std::vector<float>{30, 36, 42}));
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            tensor<float, dynamic, dynamic> b({3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto c = a * b;
            CHECK(c == tensor<float, dynamic, dynamic>({2, 2}, std::vector<float>{15, 36, 29, 71}));
        }
    }

    SUBCASE("Multiply matrix views") {
        // Subview row x column
        tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
        tensor<float, shape<2, 3>> b({1, 4, 2, 5, 3, 6});
        auto c = a.subview<1, 2>(0, 0) * b.subview<2>(0, 0);
        CHECK(c == tensor<float, shape<1, 1>>({9}));

        // Subview row x column with uncommon strides
        auto d = a.subview<shape<1, 2>, seq<1, 2>>({0, 0}) * b.subview<2>(0, 0);
        CHECK(d == tensor<float, shape<1, 1>>({13}));
    }

    SUBCASE("Multiply matrix transpose") {
        // Subview row x column
        tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
        tensor<float, shape<2, 3>> b({1, 4, 2, 5, 3, 6});
        auto c = a.transpose() * b;
        CHECK(c == tensor<float, shape<3, 3>>({17, 22, 27, 22, 29, 36, 27, 36, 45}));
    }
}

TEST_CASE("General matrix division") {
    SUBCASE("Fixed shape tensors square system") {
        tensor<float, shape<2, 2>> a({1, 3, 2, 4});
        tensor<float, shape<2, 2>> b({5, 11, 10, 22});
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(2.0).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(2.0).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(4.0).epsilon(1e-3));
    }

    SUBCASE("Fixed shape tensors overdetermined system 1D") {
        tensor<double, shape<3, 2>> a{{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}};
        tensor<double, shape<3>> b{14.0, 32.0, 50.0};
        auto c = b / a;
        CHECK(c(0) == doctest::Approx(4.0).epsilon(1e-3));
        CHECK(c(1) == doctest::Approx(5.0).epsilon(1e-3));
    }

    SUBCASE("Fixed shape tensors overdetermined system 2D") {
        tensor<float, shape<3, 2>> a{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<float, shape<3, 2>> b{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(0.0).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(0.0).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(1.0).epsilon(1e-3));
    }

    SUBCASE("Fixed shape tensors underdetermined system 1D") {
        tensor<float, shape<2, 3>> a{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<float, shape<2>> b{14.0, 32.0};
        auto c = b / a;
        CHECK(c(0) == doctest::Approx(16).epsilon(1e-4));
        CHECK(c(1) == doctest::Approx(6).epsilon(1e-4));
        CHECK(c(2) == doctest::Approx(-4).epsilon(1e-4));
    }

    SUBCASE("Fixed shape tensors underdetermined system 2D") {
        tensor<double, shape<2, 3>> a{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        tensor<double, shape<2, 2>> b{{14.0, 28.0, 32.0, 64.0}};
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(11.6666).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(4.6666).epsilon(1e-3));
        CHECK(c(2, 0) == doctest::Approx(-2.3333).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(26.6666).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(10.6666).epsilon(1e-3));
        CHECK(c(2, 1) == doctest::Approx(-5.3333).epsilon(1e-3));
    }

    SUBCASE("Dynamic shape tensors  square system") {
        tensor<float, dynamic, dynamic> a({2, 2}, std::vector<float>{1, 3, 2, 4});
        tensor<float, dynamic, dynamic> b({2, 2}, std::vector<float>{5, 11, 10, 22});
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(2.0).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(2.0).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(4.0).epsilon(1e-3));
    }

    SUBCASE("Dynamic shape tensors overdetermined system 1D") {
        tensor<double, dynamic, dynamic> a({3, 2}, std::vector<double>{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});
        tensor<double, dynamic, dynamic> b({3}, std::vector<double>{14.0, 32.0, 50.0});
        auto c = b / a;
        CHECK(c(0) == doctest::Approx(4.0).epsilon(1e-3));
        CHECK(c(1) == doctest::Approx(5.0).epsilon(1e-3));
    }

    SUBCASE("Dynamic shape tensors overdetermined system 2D") {
        tensor<float, dynamic, dynamic> a({3, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        tensor<float, dynamic, dynamic> b({3, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(0.0).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(0.0).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(1.0).epsilon(1e-3));
    }

    SUBCASE("Dynamic shape tensors underdetermined system 1D") {
        tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        tensor<float, dynamic, dynamic> b({2}, std::vector<float>{14.0, 32.0});
        auto c = b / a;
        CHECK(c(0) == doctest::Approx(16).epsilon(1e-4));
        CHECK(c(1) == doctest::Approx(6).epsilon(1e-4));
        CHECK(c(2) == doctest::Approx(-4).epsilon(1e-4));
    }

    SUBCASE("Dynamic shape tensors underdetermined system 2D") {
        tensor<double, dynamic, dynamic> a({2, 3}, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        tensor<double, dynamic, dynamic> b({2, 2}, std::vector<double>{14.0, 28.0, 32.0, 64.0});
        auto c = b / a;
        CHECK(c(0, 0) == doctest::Approx(11.6666).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(4.6666).epsilon(1e-3));
        CHECK(c(2, 0) == doctest::Approx(-2.3333).epsilon(1e-3));
        CHECK(c(0, 1) == doctest::Approx(26.6666).epsilon(1e-3));
        CHECK(c(1, 1) == doctest::Approx(10.6666).epsilon(1e-3));
        CHECK(c(2, 1) == doctest::Approx(-5.3333).epsilon(1e-3));
    }
}

TEST_CASE("Tensor Ops Type Deduction") {
    auto a = tensor<length, shape<2, 3>>::arange(length(1.0f), length(1.0f));
    auto b = tensor<length, shape<3, 2>>::arange(length(4.0f), length(1.0f));

    auto c = a.subview<2, 2>(0, 0) + b.subview<2, 2>(0, 0);
    static_assert(std::is_same_v<decltype(c), tensor<length, shape<2, 2>>>, "Type deduction failed");

    auto d = a * 2.0f;
    static_assert(std::is_same_v<decltype(d), tensor<length, shape<2, 3>>>, "Type deduction failed");

    auto e = a * b;
    static_assert(std::is_same_v<decltype(e), tensor<area, shape<2, 2>>>, "Type deduction failed");
}

// NOLINTEND