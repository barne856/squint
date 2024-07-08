#include "squint/quantity.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/tensor.hpp"

using namespace squint;

TEST_CASE("Fixed Tensor Creation and Basic Operations") {
    SUBCASE("Default constructor") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t;
        CHECK(t.rank() == 2);
        CHECK(t.size() == 6);
        CHECK(t.shape() == std::vector<std::size_t>{2, 3});
        CHECK(t.get_layout() == layout::row_major);
    }

    SUBCASE("Constructor with initializer list") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 2);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 0] == 4);
        CHECK(t[1, 1] == 5);
        CHECK(t[1, 2] == 6);
    }

    SUBCASE("Copy constructor") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t1{{1, 2, 3, 4, 5, 6}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t2(t1);
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Move constructor") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t1{{1, 2, 3, 4, 5, 6}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t2(std::move(t1));
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Assignment operator") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t1{{1, 2, 3, 4, 5, 6}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t2;
        t2 = t1;
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Move assignment operator") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t1{{1, 2, 3, 4, 5, 6}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t2;
        t2 = std::move(t1);
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }
}

TEST_CASE("Fixed Tensor Element Access") {
    fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};

    SUBCASE("Multidimensional subscript operator") {
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 2);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 0] == 4);
        CHECK(t[1, 1] == 5);
        CHECK(t[1, 2] == 6);
    }

    SUBCASE("at() method") {
        CHECK(t.at(0, 0) == 1);
        CHECK(t.at(1, 2) == 6);
    }

    SUBCASE("at() method with vector of indices") {
        CHECK(t.at_impl({0, 0}) == 1);
        CHECK(t.at_impl({1, 2}) == 6);
    }
}

TEST_CASE("Fixed Tensor Layout and Strides") {
    SUBCASE("Row-major layout") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t;
        CHECK(t.get_layout() == layout::row_major);
        CHECK(t.strides() == std::vector<std::size_t>{3, 1});
    }

    SUBCASE("Column-major layout") {
        fixed_tensor<int, layout::column_major, error_checking::disabled, 2, 3> t;
        CHECK(t.get_layout() == layout::column_major);
        CHECK(t.strides() == std::vector<std::size_t>{1, 2});
    }
}

TEST_CASE("Fixed Tensor Views") {
    fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    SUBCASE("Create view") {
        auto view = t.view();
        CHECK(view[0, 0] == 1);
        CHECK(view[2, 3] == 12);
    }

    SUBCASE("Create const view") {
        const auto &const_t = t;
        auto const_view = const_t.view();
        CHECK(const_view[0, 0] == 1);
        CHECK(const_view[2, 3] == 12);
    }

    SUBCASE("Modify through view") {
        auto view = t.view();
        view[1, 1] = 100;
        CHECK(t[1, 1] == 100);
    }

    SUBCASE("Create subview") {
        auto subview = t.subview<2, 2>(slice{0, 2}, slice{1, 2});
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 1] == 7);
    }

    SUBCASE("Modify through subview") {
        auto subview = t.subview<2, 2>(slice{0, 2}, slice{1, 2});
        subview[0, 1] = 100;
        CHECK(t[0, 2] == 100);
    }
}

TEST_CASE("Fixed Tensor Iteration") {
    fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};

    SUBCASE("Range-based for loop") {
        std::vector<int> values;
        for (const auto &value : t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }

    SUBCASE("Iterator-based loop") {
        std::vector<int> values;
        for (auto it = t.begin(); it != t.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }

    SUBCASE("Const iteration") {
        const auto &const_t = t;
        std::vector<int> values;
        for (const auto &value : const_t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }
}

TEST_CASE("Fixed Tensor Subview Iteration") {
    fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    SUBCASE("Iterate over 1x4 subviews") {
        std::vector<std::vector<int>> subview_values;
        for (const auto &subview : t.subviews<1, 4>()) {
            std::vector<int> values;
            for (const auto &value : subview) {
                values.push_back(value);
            }
            subview_values.push_back(values);
        }
        CHECK(subview_values == std::vector<std::vector<int>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    }
}

TEST_CASE("Dynamic Tensor Creation and Basic Operations") {
    SUBCASE("Constructor with shape") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});
        CHECK(t.rank() == 2);
        CHECK(t.size() == 6);
        CHECK(t.shape() == std::vector<std::size_t>{2, 3});
        CHECK(t.get_layout() == layout::column_major);
    }

    SUBCASE("Constructor with shape and layout") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, layout::row_major);
        CHECK(t.get_layout() == layout::row_major);
    }

    SUBCASE("Copy constructor") {
        dynamic_tensor<int, error_checking::disabled> t1({2, 3});
        t1[0, 0] = 1;
        t1[1, 2] = 6;
        dynamic_tensor<int, error_checking::disabled> t2(t1);
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Move constructor") {
        dynamic_tensor<int, error_checking::disabled> t1({2, 3});
        t1[0, 0] = 1;
        t1[1, 2] = 6;
        dynamic_tensor<int, error_checking::disabled> t2(std::move(t1));
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Assignment operator") {
        dynamic_tensor<int, error_checking::disabled> t1({2, 3});
        t1[0, 0] = 1;
        t1[1, 2] = 6;
        dynamic_tensor<int, error_checking::disabled> t2({2, 3});
        t2 = t1;
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }

    SUBCASE("Move assignment operator") {
        dynamic_tensor<int, error_checking::disabled> t1({2, 3});
        t1[0, 0] = 1;
        t1[1, 2] = 6;
        dynamic_tensor<int, error_checking::disabled> t2({2, 3});
        t2 = std::move(t1);
        CHECK(t2[0, 0] == 1);
        CHECK(t2[1, 2] == 6);
    }
}

TEST_CASE("Dynamic Tensor Element Access") {
    dynamic_tensor<int, error_checking::disabled> t({2, 3});
    t[0, 0] = 1;
    t[0, 1] = 2;
    t[0, 2] = 3;
    t[1, 0] = 4;
    t[1, 1] = 5;
    t[1, 2] = 6;

    SUBCASE("Multidimensional subscript operator") {
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 2);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 0] == 4);
        CHECK(t[1, 1] == 5);
        CHECK(t[1, 2] == 6);
    }

    SUBCASE("at() method") {
        CHECK(t.at(0, 0) == 1);
        CHECK(t.at(1, 2) == 6);
    }

    SUBCASE("at() method with vector of indices") {
        CHECK(t.at({0, 0}) == 1);
        CHECK(t.at({1, 2}) == 6);
    }
}

TEST_CASE("Dynamic Tensor Layout and Strides") {
    SUBCASE("Row-major layout") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, layout::row_major);
        CHECK(t.get_layout() == layout::row_major);
        CHECK(t.strides() == std::vector<std::size_t>{3, 1});
    }

    SUBCASE("Column-major layout") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, layout::column_major);
        CHECK(t.get_layout() == layout::column_major);
        CHECK(t.strides() == std::vector<std::size_t>{1, 2});
    }
}

TEST_CASE("Dynamic Tensor Views") {
    dynamic_tensor<int, error_checking::disabled> t({3, 4});
    for (int i = 0; i < 12; ++i) {
        t[i / 4, i % 4] = i + 1;
    }

    SUBCASE("Create view") {
        auto view = t.view();
        CHECK(view[0, 0] == 1);
        CHECK(view[2, 3] == 12);
    }

    SUBCASE("Create const view") {
        const auto &const_t = t;
        auto const_view = const_t.view();
        CHECK(const_view[0, 0] == 1);
        CHECK(const_view[2, 3] == 12);
    }

    SUBCASE("Modify through view") {
        auto view = t.view();
        view[1, 1] = 100;
        CHECK(t[1, 1] == 100);
    }

    SUBCASE("Create subview") {
        auto subview = t.subview(slice{0, 2}, slice{1, 2});
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 1] == 7);
    }

    SUBCASE("Modify through subview") {
        auto subview = t.subview(slice{0, 2}, slice{1, 2});
        subview[0, 1] = 100;
        CHECK(t[0, 2] == 100);
    }

    SUBCASE("Subview with vector of slices") {
        std::vector<slice> slices = {slice{0, 2}, slice{1, 2}};
        auto subview = t.subview(slices);
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 1] == 7);
    }
}

TEST_CASE("Dynamic Tensor Iteration") {
    dynamic_tensor<int, error_checking::disabled> t({2, 3});
    t[0, 0] = 1;
    t[0, 1] = 2;
    t[0, 2] = 3;
    t[1, 0] = 4;
    t[1, 1] = 5;
    t[1, 2] = 6;

    SUBCASE("Range-based for loop") {
        std::vector<int> values;
        for (const auto &value : t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }

    SUBCASE("Iterator-based loop") {
        std::vector<int> values;
        for (auto it = t.begin(); it != t.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }

    SUBCASE("Const iteration") {
        const auto &const_t = t;
        std::vector<int> values;
        for (const auto &value : const_t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }
}

TEST_CASE("Dynamic Tensor Subview Iteration") {
    dynamic_tensor<int, error_checking::disabled> t({3, 4});
    for (int i = 0; i < 12; ++i) {
        t[i / 4, i % 4] = i + 1;
    }

    SUBCASE("Iterate over 1x4 subviews") {
        std::vector<std::vector<int>> subview_values;
        for (const auto &subview : t.subviews({1, 4})) {
            std::vector<int> values;
            for (const auto &value : subview) {
                values.push_back(value);
            }
            subview_values.push_back(values);
        }
        CHECK(subview_values == std::vector<std::vector<int>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    }
}

TEST_CASE("Tensor Reshape") {
    SUBCASE("Dynamic tensor reshape (column major)") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});
        for (int i = 0; i < 6; ++i) {
            t[i / 3, i % 3] = i + 1;
        }

        t.reshape({3, 2});
        CHECK(t.shape() == std::vector<std::size_t>{3, 2});
        CHECK(t[0, 0] == 1);
        CHECK(t[1, 1] == 3);
        CHECK(t[2, 1] == 6);

        CHECK_THROWS_AS(t.reshape({2, 2}), std::invalid_argument);
    }

    SUBCASE("Dynamic tensor reshape (row major)") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, layout::row_major);
        for (int i = 0; i < 6; ++i) {
            t[i / 3, i % 3] = i + 1;
        }

        t.reshape({3, 2});
        CHECK(t.shape() == std::vector<std::size_t>{3, 2});
        CHECK(t[0, 0] == 1);
        CHECK(t[1, 1] == 4);
        CHECK(t[2, 1] == 6);

        CHECK_THROWS_AS(t.reshape({2, 2}), std::invalid_argument);
    }
}

TEST_CASE("Tensor Concepts") {
    SUBCASE("fixed_shape_tensor concept") {
        CHECK(fixed_shape_tensor<fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3>>);
        CHECK_FALSE(fixed_shape_tensor<dynamic_tensor<int, error_checking::disabled>>);
    }

    SUBCASE("dynamic_shape_tensor concept") {
        CHECK(dynamic_shape_tensor<dynamic_tensor<int, error_checking::disabled>>);
        CHECK_FALSE(dynamic_shape_tensor<fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3>>);
    }
}

TEST_CASE("Tensor Stream Output") {
    SUBCASE("Fixed tensor output") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        std::ostringstream oss;
        oss << t;
        CHECK(oss.str() == "Tensor(shape=[2, 3], data=[[1, 2, 3], [4, 5, 6]])");
    }

    SUBCASE("Dynamic tensor output") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});
        t[0, 0] = 1;
        t[0, 1] = 2;
        t[0, 2] = 3;
        t[1, 0] = 4;
        t[1, 1] = 5;
        t[1, 2] = 6;
        std::ostringstream oss;
        oss << t;
        CHECK(oss.str() == "Tensor(shape=[2, 3], data=[[1, 2, 3], [4, 5, 6]])");
    }
}

TEST_CASE("Tensor View Edge Cases") {
    SUBCASE("Empty fixed tensor view") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 0, 3> t;
        auto view = t.view();
        CHECK(view.size() == 0);
        CHECK(view.shape() == std::vector<std::size_t>{0, 3});
    }

    SUBCASE("Empty dynamic tensor view") {
        dynamic_tensor<int, error_checking::disabled> t({0, 3});
        auto view = t.view();
        CHECK(view.size() == 0);
        CHECK(view.shape() == std::vector<std::size_t>{0, 3});
    }

    SUBCASE("Fixed tensor view with single element") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 1, 1> t{{42}};
        auto view = t.view();
        CHECK(view.size() == 1);
        CHECK(view[0, 0] == 42);
    }

    SUBCASE("Dynamic tensor view with single element") {
        dynamic_tensor<int, error_checking::disabled> t({1, 1});
        t[0, 0] = 42;
        auto view = t.view();
        CHECK(view.size() == 1);
        CHECK(view[0, 0] == 42);
    }
}

TEST_CASE("Tensor View Const Correctness") {
    SUBCASE("Fixed tensor const view") {
        const fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        auto view = t.view();
        CHECK(view[0, 0] == 1);
        // The following line should not compile:
        // view[0, 0] = 10;
    }

    SUBCASE("Dynamic tensor const view") {
        const dynamic_tensor<int, error_checking::disabled> t({2, 3});
        auto view = t.view();
        CHECK(view[0, 0] == 0);
        // The following line should not compile:
        // view[0, 0] = 10;
    }
}

TEST_CASE("Tensor View Slicing") {
    SUBCASE("Fixed tensor slicing") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t;
        for (int i = 0; i < 12; ++i) {
            t[i / 4, i % 4] = i + 1;
        }

        auto view = t.view();
        auto subview = view.subview<2, 3>(slice{0, 2}, slice{1, 3});
        CHECK(subview.shape() == std::vector<std::size_t>{2, 3});
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 2] == 8);
    }

    SUBCASE("Dynamic tensor slicing") {
        dynamic_tensor<int, error_checking::disabled> t({3, 4});
        for (int i = 0; i < 12; ++i) {
            t[i / 4, i % 4] = i + 1;
        }

        auto view = t.view();
        auto subview = view.subview(slice{0, 2}, slice{1, 3});
        CHECK(subview.shape() == std::vector<std::size_t>{2, 3});
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 2] == 8);
    }
}

TEST_CASE("Tensor View Iterator") {
    SUBCASE("Fixed tensor view iterator") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        auto view = t.view();
        std::vector<int> values;
        for (auto it = view.begin(); it != view.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }

    SUBCASE("Dynamic tensor view iterator") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});
        t[0, 0] = 1;
        t[0, 1] = 2;
        t[0, 2] = 3;
        t[1, 0] = 4;
        t[1, 1] = 5;
        t[1, 2] = 6;
        auto view = t.view();
        std::vector<int> values;
        for (auto it = view.begin(); it != view.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<int>{1, 2, 3, 4, 5, 6});
    }
}

TEST_CASE("Tensor View Subview Iterator") {
    SUBCASE("Fixed tensor view subview iterator") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t;
        for (int i = 0; i < 12; ++i) {
            t[i / 4, i % 4] = i + 1;
        }
        auto view = t.view();
        std::vector<std::vector<int>> subview_values;
        for (const auto &subview : view.subviews<1, 4>()) {
            std::vector<int> values;
            for (const auto &value : subview) {
                values.push_back(value);
            }
            subview_values.push_back(values);
        }
        CHECK(subview_values == std::vector<std::vector<int>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    }

    SUBCASE("Dynamic tensor view subview iterator") {
        dynamic_tensor<int, error_checking::disabled> t({3, 4});
        for (int i = 0; i < 12; ++i) {
            t[i / 4, i % 4] = i + 1;
        }
        auto view = t.view();
        std::vector<std::vector<int>> subview_values;
        for (const auto &subview : view.subviews({1, 4})) {
            std::vector<int> values;
            for (const auto &value : subview) {
                values.push_back(value);
            }
            subview_values.push_back(values);
        }
        CHECK(subview_values == std::vector<std::vector<int>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    }
}

TEST_CASE("Fixed Tensor with Error Checking") {
    fixed_tensor<int, layout::row_major, error_checking::enabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};

    SUBCASE("Out of bounds access") {
        CHECK_THROWS_AS(t.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(t.at(0, 3), std::out_of_range);
    }

    // invalid number of indices should not compile
    // t.at(0);
    // t.at(0, 0, 0);

    SUBCASE("Subview out of bounds") {
        CHECK_THROWS_AS((t.subview<2, 2>(slice{1, 2}, slice{0, 2})), std::out_of_range);
        CHECK_THROWS_AS((t.subview<2, 2>(slice{0, 2}, slice{2, 2})), std::out_of_range);
    }
}

TEST_CASE("Dynamic Tensor with Error Checking") {
    dynamic_tensor<int, error_checking::enabled> t({2, 3});

    SUBCASE("Out of bounds access") {
        CHECK_THROWS_AS(t.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(t.at(0, 3), std::out_of_range);
    }

    SUBCASE("Invalid number of indices") {
        CHECK_THROWS_AS(t.at(0), std::out_of_range);
        CHECK_THROWS_AS(t.at(0, 0, 0), std::out_of_range);
    }

    SUBCASE("Subview out of bounds") {
        CHECK_THROWS_AS(t.subview(slice{2, 1}, slice{0, 3}), std::out_of_range);
        CHECK_THROWS_AS(t.subview(slice{0, 2}, slice{3, 1}), std::out_of_range);
    }
}

TEST_CASE("Tensor View Error Checking") {
    SUBCASE("Fixed tensor view with error checking") {
        fixed_tensor<int, layout::row_major, error_checking::enabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        auto view = t.view();

        CHECK_THROWS_AS(view.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(view.at(0, 3), std::out_of_range);
        CHECK_THROWS_AS((view.subview<2, 2>(slice{1, 2}, slice{0, 2})), std::out_of_range);
    }

    SUBCASE("Dynamic tensor view with error checking") {
        dynamic_tensor<int, error_checking::enabled> t({2, 3});
        auto view = t.view();

        CHECK_THROWS_AS(view.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(view.at(0, 3), std::out_of_range);
        CHECK_THROWS_AS(view.subview(slice{2, 1}, slice{0, 3}), std::out_of_range);
    }
}

TEST_CASE("Error Checking Disabled") {
    SUBCASE("Fixed tensor without error checking") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};

        CHECK_NOTHROW(t.at(2, 0)); // This would be out of bounds, but no check is performed
        CHECK_NOTHROW(t.at(0, 3)); // This would be out of bounds, but no check is performed
        CHECK_NOTHROW(t.subview<2, 2>(slice{1, 2}, slice{0, 2})); // This would be invalid, but no check is performed
    }

    SUBCASE("Dynamic tensor without error checking") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});

        CHECK_NOTHROW(t.at(2, 0));                          // This would be out of bounds, but no check is performed
        CHECK_NOTHROW(t.at(0, 3));                          // This would be out of bounds, but no check is performed
        CHECK_NOTHROW(t.subview(slice{2, 1}, slice{0, 3})); // This would be invalid, but no check is performed
    }
}