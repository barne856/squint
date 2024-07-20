#include "squint/quantity.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/geometry.hpp"
#include <cmath>

using namespace squint;

auto pi = squint::constants::math_constants<float>::pi;

TEST_CASE("Translation") {
    auto matrix = mat4::eye();
    auto translation = vec3_t<units::length>{{units::length(1.0F), units::length(2.0F), units::length(3.0F)}};

    SUBCASE("Default unit length") {
        translate(matrix, translation);
        CHECK(matrix[0, 3] == doctest::Approx(1.0F));
        CHECK(matrix[1, 3] == doctest::Approx(2.0F));
        CHECK(matrix[2, 3] == doctest::Approx(3.0F));
    }

    SUBCASE("Custom unit length") {
        auto unit_length = units::length(2.0F);
        translate(matrix, translation, unit_length);
        CHECK(matrix[0, 3] == doctest::Approx(0.5F));
        CHECK(matrix[1, 3] == doctest::Approx(1.0F));
        CHECK(matrix[2, 3] == doctest::Approx(1.5F));
    }
}

TEST_CASE("Rotation") {
    auto matrix = mat4::eye();

    float angle = static_cast<float>(pi) / 2.0F; // 90 degrees

    SUBCASE("Rotation around X-axis") {
        auto axis = vec3{{1.0F, 0.0F, 0.0F}};
        rotate(matrix, angle, axis);
        CHECK(matrix[1, 1] == doctest::Approx(0.0F).epsilon(0.001F));
        CHECK(matrix[1, 2] == doctest::Approx(-1.0F).epsilon(0.001F));
        CHECK(matrix[2, 1] == doctest::Approx(1.0F).epsilon(0.001F));
        CHECK(matrix[2, 2] == doctest::Approx(0.0F).epsilon(0.001F));
    }

    SUBCASE("Rotation around arbitrary axis") {
        auto axis = vec3{{1.0F, 1.0F, 1.0F}};
        rotate(matrix, angle, axis);
        CHECK(matrix[0, 0] == doctest::Approx(0.333333F).epsilon(0.001F));
        CHECK(matrix[1, 1] == doctest::Approx(0.333333F).epsilon(0.001F));
        CHECK(matrix[2, 2] == doctest::Approx(0.333333F).epsilon(0.001F));
    }
}

TEST_CASE("Scale") {
    auto matrix = mat4::eye();
    auto scale_factors = vec3{{2.0F, 3.0F, 4.0F}};

    scale(matrix, scale_factors);
    CHECK(matrix[0, 0] == doctest::Approx(2.0F));
    CHECK(matrix[1, 1] == doctest::Approx(3.0F));
    CHECK(matrix[2, 2] == doctest::Approx(4.0F));
}

TEST_CASE("Orthographic Projection") {
    auto left = units::length(-1.0F);
    auto right = units::length(1.0F);
    auto bottom = units::length(-1.0F);
    auto top = units::length(1.0F);
    auto near_plane = units::length(0.1F);
    auto far_plane = units::length(100.0F);

    SUBCASE("Default unit length") {
        auto result = ortho(left, right, bottom, top, near_plane, far_plane);
        CHECK(result[0, 0] == doctest::Approx(1.0F));
        CHECK(result[1, 1] == doctest::Approx(1.0F));
        CHECK(result[2, 2] == doctest::Approx(0.0100100F).epsilon(0.0001F));
        CHECK(result[0, 3] == doctest::Approx(0.0F));
        CHECK(result[1, 3] == doctest::Approx(0.0F));
        CHECK(result[2, 3] == doctest::Approx(-0.001001F).epsilon(0.0001F));
        CHECK(result[3, 3] == doctest::Approx(1.0F));
    }

    SUBCASE("Custom unit length") {
        auto unit_length = units::length(2.0F);
        auto result = ortho(left, right, bottom, top, near_plane, far_plane, unit_length);
        CHECK(result[0, 0] == doctest::Approx(1.0F * 2.0F));
        CHECK(result[1, 1] == doctest::Approx(1.0F * 2.0F));
        CHECK(result[2, 2] == doctest::Approx(0.0100100F * 2.0F).epsilon(0.0001F));
        CHECK(result[0, 3] == doctest::Approx(0.0F));
        CHECK(result[1, 3] == doctest::Approx(0.0F));
        CHECK(result[2, 3] == doctest::Approx(-0.001001F).epsilon(0.0001F));
    }
}

TEST_CASE("Perspective Projection") {
    float fovy = static_cast<float>(pi) / 4.0F; // 45 degrees
    float aspect = 16.0F / 9.0F;
    auto near_plane = units::length(0.1F);
    auto far_plane = units::length(100.0F);

    SUBCASE("Default unit length") {
        auto result = perspective(fovy, aspect, near_plane, far_plane);
        CHECK(result[0, 0] == doctest::Approx(1.3578979F).epsilon(0.0001F));
        CHECK(result[1, 1] == doctest::Approx(2.4142134F).epsilon(0.0001F));
        CHECK(result[2, 2] == doctest::Approx(-1.001001F).epsilon(0.0001F));
        CHECK(result[2, 3] == doctest::Approx(-0.1001002F).epsilon(0.0001F));
        CHECK(result[3, 2] == doctest::Approx(-1.0F));
    }

    SUBCASE("Custom unit length") {
        auto unit_length = units::length(2.0F);
        auto result = perspective(fovy, aspect, near_plane, far_plane, unit_length);
        CHECK(result[0, 0] == doctest::Approx(1.3578979F).epsilon(0.0001F));
        CHECK(result[1, 1] == doctest::Approx(2.4142134F).epsilon(0.0001F));
        CHECK(result[2, 2] == doctest::Approx(-1.001001F).epsilon(0.0001F));
        CHECK(result[2, 3] == doctest::Approx(-0.0500500F).epsilon(0.0001F));
        CHECK(result[3, 2] == doctest::Approx(-1.0F));
    }
}