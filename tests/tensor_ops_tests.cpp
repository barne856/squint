// NOLINTBEGIN
#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
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
            CHECK(c(0) == 32);
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
}

// NOLINTEND