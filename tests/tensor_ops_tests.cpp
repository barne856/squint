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
#ifdef SQUINT_USE_CUDA
        auto a_device = a.to_device();
        auto b_device = b.to_device();
#endif

        SUBCASE("Addition") {
            auto c = a + b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i, j) + b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = a_device + b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(a(i, j) + b(i, j)));
                }
            }
#endif
        }

        SUBCASE("Subtraction") {
            auto c = a - b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i, j) - b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = a_device - b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(1));
                }
            }
#endif
        }

        SUBCASE("Unary negation") {
            auto c = -a;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(-a(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = -a_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(-a(i, j)));
                }
            }
#endif
        }

        SUBCASE("In-place addition") {
            a += b;
            CHECK(a(0, 0) == doctest::Approx(3));
            CHECK(a(1, 0) == doctest::Approx(9));
            CHECK(a(0, 1) == doctest::Approx(5));
            CHECK(a(1, 1) == doctest::Approx(11));
            CHECK(a(0, 2) == doctest::Approx(7));
            CHECK(a(1, 2) == doctest::Approx(13));
#ifdef SQUINT_USE_CUDA
            a_device += b_device;
            auto a_host = a_device.to_host();
            CHECK(a_host(0, 0) == doctest::Approx(3));
            CHECK(a_host(1, 0) == doctest::Approx(9));
            CHECK(a_host(0, 1) == doctest::Approx(5));
            CHECK(a_host(1, 1) == doctest::Approx(11));
            CHECK(a_host(0, 2) == doctest::Approx(7));
            CHECK(a_host(1, 2) == doctest::Approx(13));
#endif
        }

        SUBCASE("In-place subtraction") {
            a -= b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx(-1));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device -= b_device;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx(1));
                }
            }
#endif
        }
    }

    SUBCASE("Dynamic shape tensors") {
        tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        tensor<float, dynamic, dynamic> b({2, 3}, std::vector<float>{2, 5, 3, 6, 4, 7});
#ifdef SQUINT_USE_CUDA
        auto a_device = a.to_device();
        auto b_device = b.to_device();
#endif

        SUBCASE("Addition") {
            auto c = a + b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i, j) + b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = a_device + b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(a(i, j) + b(i, j)));
                }
            }
#endif
        }

        SUBCASE("Subtraction") {
            auto c = a - b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i, j) - b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = a_device - b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(1));
                }
            }
#endif
        }

        SUBCASE("Unary negation") {
            auto c = -a;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(-a(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = -a_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(-a(i, j)));
                }
            }
#endif
        }

        SUBCASE("In-place addition") {
            a += b;
            CHECK(a(0, 0) == doctest::Approx(3));
            CHECK(a(1, 0) == doctest::Approx(9));
            CHECK(a(0, 1) == doctest::Approx(5));
            CHECK(a(1, 1) == doctest::Approx(11));
            CHECK(a(0, 2) == doctest::Approx(7));
            CHECK(a(1, 2) == doctest::Approx(13));
#ifdef SQUINT_USE_CUDA
            a_device += b_device;
            auto a_host = a_device.to_host();
            CHECK(a_host(0, 0) == doctest::Approx(3));
            CHECK(a_host(1, 0) == doctest::Approx(9));
            CHECK(a_host(0, 1) == doctest::Approx(5));
            CHECK(a_host(1, 1) == doctest::Approx(11));
            CHECK(a_host(0, 2) == doctest::Approx(7));
            CHECK(a_host(1, 2) == doctest::Approx(13));
#endif
        }

        SUBCASE("In-place subtraction") {
            a -= b;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx(-1));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device -= b_device;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx(1));
                }
            }
#endif
        }
    }
}

TEST_CASE("Scalar operations") {
    SUBCASE("Fixed shape tensors") {
        tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
#ifdef SQUINT_USE_CUDA
        auto a_device = a.to_device();
#endif

        SUBCASE("Scalar multiplication") {
            auto b = a * 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b(i, j) == doctest::Approx(2 * a(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto b_device = a_device * 2.0f;
            auto b_host = b_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b_host(i, j) == doctest::Approx(2 * a(i, j)));
                }
            }
#endif

            auto c = 2.0f * a;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = 2.0f * a_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(b(i, j)));
                }
            }
#endif
        }

        SUBCASE("Scalar division") {
            auto b = a / 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b(i, j) == doctest::Approx(a(i, j) / 2.0f));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto b_device = a_device / 2.0f;
            auto b_host = b_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b_host(i, j) == doctest::Approx(a(i, j) / 2.0f));
                }
            }
#endif
        }

        SUBCASE("In-place scalar multiplication") {
            a *= 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx(2 * (1 + i * 3 + j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device *= 2.0f;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx(2 * (1 + i * 3 + j)));
                }
            }
#endif
        }

        SUBCASE("In-place scalar division") {
            a /= 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx((1 + i * 3 + j) / 2.0f));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device /= 2.0f;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx((1 + i * 3 + j) / 2.0f));
                }
            }
#endif
        }
    }

    SUBCASE("Dynamic shape tensors") {
        tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
#ifdef SQUINT_USE_CUDA
        auto a_device = a.to_device();
#endif

        SUBCASE("Scalar multiplication") {
            auto b = a * 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b(i, j) == doctest::Approx(2 * a(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto b_device = a_device * 2.0f;
            auto b_host = b_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b_host(i, j) == doctest::Approx(2 * a(i, j)));
                }
            }
#endif

            auto c = 2.0f * a;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(b(i, j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto c_device = 2.0f * a_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(b(i, j)));
                }
            }
#endif
        }

        SUBCASE("Scalar division") {
            auto b = a / 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b(i, j) == doctest::Approx(a(i, j) / 2.0f));
                }
            }
#ifdef SQUINT_USE_CUDA
            auto b_device = a_device / 2.0f;
            auto b_host = b_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(b_host(i, j) == doctest::Approx(a(i, j) / 2.0f));
                }
            }
#endif
        }

        SUBCASE("In-place scalar multiplication") {
            a *= 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx(2 * (1 + i * 3 + j)));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device *= 2.0f;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx(2 * (1 + i * 3 + j)));
                }
            }
#endif
        }

        SUBCASE("In-place scalar division") {
            a /= 2.0f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a(i, j) == doctest::Approx((1 + i * 3 + j) / 2.0f));
                }
            }
#ifdef SQUINT_USE_CUDA
            a_device /= 2.0f;
            auto a_host = a_device.to_host();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(a_host(i, j) == doctest::Approx((1 + i * 3 + j) / 2.0f));
                }
            }
#endif
        }
    }
}

TEST_CASE("Matrix multiplication") {
    SUBCASE("Fixed shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto c = a.transpose() * b;
            CHECK(c(0, 0) == doctest::Approx(32));
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto c = a * b.transpose();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i) * b(j)));
                }
            }
        }

        SUBCASE("Vector times matrix") {
            tensor<float, shape<2>> a({1, 2});
            tensor<float, shape<2, 3>> b({1, 4, 2, 5, 3, 6});
            auto c = a.transpose() * b;
            for (int j = 0; j < 3; ++j) {
                CHECK(c(0, j) == doctest::Approx(a(0) * b(0, j) + a(1) * b(1, j)));
            }
        }

        SUBCASE("Matrix times vector") {
            tensor<float, shape<3, 3>> a({1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, shape<3>> b({1, 2, 3});
            auto c = a * b;
            for (int i = 0; i < 3; ++i) {
                float expected = 0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(i, j) * b(j);
                }
                CHECK(c(i) == doctest::Approx(expected));
            }
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
            tensor<float, shape<3, 2>> b({1, 4, 2, 5, 3, 6});
            auto c = a * b;
            CHECK(c(0, 0) == doctest::Approx(15));
            CHECK(c(0, 1) == doctest::Approx(29));
            CHECK(c(1, 0) == doctest::Approx(36));
            CHECK(c(1, 1) == doctest::Approx(71));
        }
    }

    SUBCASE("Dynamic shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto c = a.transpose() * b;
            CHECK(c(0, 0) == doctest::Approx(32));
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto c = a * b.transpose();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c(i, j) == doctest::Approx(a(i) * b(j)));
                }
            }
        }

        SUBCASE("Vector times matrix") {
            tensor<float, dynamic, dynamic> a({2}, std::vector<float>{1, 2});
            tensor<float, dynamic, dynamic> b({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto c = a.transpose() * b;
            for (int j = 0; j < 3; ++j) {
                CHECK(c(0, j) == doctest::Approx(a(0) * b(0, j) + a(1) * b(1, j)));
            }
        }

        SUBCASE("Matrix times vector") {
            tensor<float, dynamic, dynamic> a({3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{1, 2, 3});
            auto c = a * b;
            for (int i = 0; i < 3; ++i) {
                float expected = 0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(i, j) * b(j);
                }
                CHECK(c(i, 0) == doctest::Approx(expected));
            }
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            tensor<float, dynamic, dynamic> b({3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto c = a * b;
            CHECK(c(0, 0) == doctest::Approx(15));
            CHECK(c(0, 1) == doctest::Approx(29));
            CHECK(c(1, 0) == doctest::Approx(36));
            CHECK(c(1, 1) == doctest::Approx(71));
        }
    }
}
#ifdef SQUINT_USE_CUDA
TEST_CASE("Matrix multiplication device") {
    SUBCASE("Fixed shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device.transpose() * b_device;
            auto c_host = c_device.to_host();
            CHECK(c_host(0, 0) == doctest::Approx(32));
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, shape<3>> a({1, 2, 3});
            tensor<float, shape<3>> b({4, 5, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device.transpose();
            auto c_host = c_device.to_host();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(a(i) * b(j)));
                }
            }
        }

        SUBCASE("Vector times matrix") {
            tensor<float, shape<2>> a({1, 2});
            tensor<float, shape<2, 3>> b({1, 4, 2, 5, 3, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device.transpose() * b_device;
            auto c_host = c_device.to_host();
            for (int j = 0; j < 3; ++j) {
                CHECK(c_host(0, j) == doctest::Approx(a(0) * b(0, j) + a(1) * b(1, j)));
            }
        }

        SUBCASE("Matrix times vector") {
            tensor<float, shape<3, 3>> a({1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, shape<3>> b({1, 2, 3});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 3; ++i) {
                float expected = 0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(i, j) * b(j);
                }
                CHECK(c_host(i, 0) == doctest::Approx(expected));
            }
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, shape<2, 3>> a({1, 4, 2, 5, 3, 6});
            tensor<float, shape<3, 2>> b({1, 4, 2, 5, 3, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device;
            auto c_host = c_device.to_host();
            CHECK(c_host(0, 0) == doctest::Approx(15));
            CHECK(c_host(0, 1) == doctest::Approx(29));
            CHECK(c_host(1, 0) == doctest::Approx(36));
            CHECK(c_host(1, 1) == doctest::Approx(71));
        }
    }

    SUBCASE("Dynamic shape tensors") {
        SUBCASE("Inner product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device.transpose() * b_device;
            auto c_host = c_device.to_host();
            CHECK(c_host(0, 0) == doctest::Approx(32));
        }

        SUBCASE("Outer product of vectors") {
            tensor<float, dynamic, dynamic> a({3}, std::vector<float>{1, 2, 3});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{4, 5, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device.transpose();
            auto c_host = c_device.to_host();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK(c_host(i, j) == doctest::Approx(a(i) * b(j)));
                }
            }
        }

        SUBCASE("Vector times matrix") {
            tensor<float, dynamic, dynamic> a({2}, std::vector<float>{1, 2});
            tensor<float, dynamic, dynamic> b({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device.transpose() * b_device;
            auto c_host = c_device.to_host();
            for (int j = 0; j < 3; ++j) {
                CHECK(c_host(0, j) == doctest::Approx(a(0) * b(0, j) + a(1) * b(1, j)));
            }
        }

        SUBCASE("Matrix times vector") {
            tensor<float, dynamic, dynamic> a({3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
            tensor<float, dynamic, dynamic> b({3}, std::vector<float>{1, 2, 3});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device;
            auto c_host = c_device.to_host();
            for (int i = 0; i < 3; ++i) {
                float expected = 0;
                for (int j = 0; j < 3; ++j) {
                    expected += a(i, j) * b(j);
                }
                CHECK(c_host(i, 0) == doctest::Approx(expected));
            }
        }

        SUBCASE("Matrix times matrix") {
            tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
            tensor<float, dynamic, dynamic> b({3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6});
            auto a_device = a.to_device();
            auto b_device = b.to_device();
            auto c_device = a_device * b_device;
            auto c_host = c_device.to_host();
            CHECK(c_host(0, 0) == doctest::Approx(15));
            CHECK(c_host(0, 1) == doctest::Approx(29));
            CHECK(c_host(1, 0) == doctest::Approx(36));
            CHECK(c_host(1, 1) == doctest::Approx(71));
        }
    }
}
#endif

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
        CHECK(c(0, 0) == doctest::Approx(4.0).epsilon(1e-3));
        CHECK(c(1, 0) == doctest::Approx(5.0).epsilon(1e-3));
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
        tensor<float, dynamic, dynamic> b({2, 1}, std::vector<float>{14.0, 32.0});
        auto c = b / a;
        auto index = std::vector<std::size_t>(c.shape().size(), 0);
        index[0] = 0;
        CHECK(c[index] == doctest::Approx(16).epsilon(1e-4));
        index[0] = 1;
        CHECK(c[index] == doctest::Approx(6).epsilon(1e-4));
        index[0] = 2;
        CHECK(c[index] == doctest::Approx(-4).epsilon(1e-4));
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

    tensor<squint::duration, shape<2, 2>> f({squint::duration(1), squint::duration(3), squint::duration(2), squint::duration(4)});
    tensor<length, shape<2, 2>> g({length(5), length(11), length(10), length(22)});
    auto h = g / f;
    static_assert(std::is_same_v<decltype(h), tensor<velocity, shape<2, 2>>>, "Type deduction failed");
}

// NOLINTEND