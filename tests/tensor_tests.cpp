#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/tensor.hpp"

using namespace squint;

TEST_CASE("Fixed Tensor Sizeof Test") {
    SUBCASE("Mat4") {
        auto A = mat4::eye();
        CHECK(sizeof(A) == 4 * 4 * sizeof(float));
    }
}

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

    SUBCASE("Constructor with value") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t(3);
        CHECK(t[0, 0] == 3);
        CHECK(t[0, 1] == 3);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 0] == 3);
        CHECK(t[1, 1] == 3);
        CHECK(t[1, 2] == 3);
    }

    SUBCASE("Constructor with tensor block") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 1> t_block{{1, 2}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{t_block};
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 1);
        CHECK(t[0, 2] == 1);
        CHECK(t[1, 0] == 2);
        CHECK(t[1, 1] == 2);
        CHECK(t[1, 2] == 2);
    }

    SUBCASE("Constructor with array of blocks") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 1> t_block1{{1, 2}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 1> t_block2{{3, 4}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 1> t_block3{{5, 6}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{
            std::array{t_block1, t_block2, t_block3}};
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 3);
        CHECK(t[0, 2] == 5);
        CHECK(t[1, 0] == 2);
        CHECK(t[1, 1] == 4);
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

    SUBCASE("Assignment from const") {
        const fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t1{{1, 2, 3, 4, 5, 6}};
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
        auto subview = t.subview<2, 2>(0,1);
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 1] == 7);
    }

    SUBCASE("Modify through subview") {
        auto subview = t.subview<2, 2>(0,1);
        subview[0, 1] = 100;
        CHECK(t[0, 2] == 100);
    }

    SUBCASE("Assign from const tensor") {
        const fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> const_tens{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
        fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t;
        t.view() = const_tens;
        CHECK(t[0, 0] == 1);
        CHECK(t[2, 3] == 12);
    }

    SUBCASE("Assign from const view") {
        const fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> const_tens{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
        const auto const_view = const_tens.view();
        fixed_tensor<int, layout::row_major, error_checking::disabled, 3, 4> t;
        t.view() = const_view;
        CHECK(t[0, 0] == 1);
        CHECK(t[2, 3] == 12);
    }
}

TEST_CASE("Fixed Tensor Iteration") {
    fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};

    SUBCASE("Range-based for loop") {
        std::vector<int> values;
        for (const auto &value : t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("Iterator-based loop") {
        std::vector<int> values;
        for (auto it = t.begin(); it != t.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<int>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("Const iteration") {
        const auto &const_t = t;
        std::vector<int> values;
        for (const auto &value : const_t) {
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("Non const iteration") {
        auto &non_const_t = t;
        std::vector<int> values;
        for (auto &value : non_const_t) {
            value += 1;
            values.push_back(value);
        }
        CHECK(values == std::vector<int>{2, 5, 3, 6, 4, 7});
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

    SUBCASE("Constructor with vector of elements") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, std::vector{1, 2, 3, 4, 5, 6});
        CHECK(t[0, 0] == 1);
        CHECK(t[1, 0] == 2);
        CHECK(t[0, 1] == 3);
        CHECK(t[1, 1] == 4);
        CHECK(t[0, 2] == 5);
        CHECK(t[1, 2] == 6);
    }

    SUBCASE("Constructor with value") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, 3);
        CHECK(t[0, 0] == 3);
        CHECK(t[0, 1] == 3);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 0] == 3);
        CHECK(t[1, 1] == 3);
        CHECK(t[1, 2] == 3);
    }

    SUBCASE("Constructor with tensor block") {
        dynamic_tensor<int, error_checking::disabled> t_block({2, 1});
        t_block[0, 0] = 1;
        t_block[1, 0] = 2;
        dynamic_tensor<int, error_checking::disabled> t({2, 2}, t_block);
        CHECK(t[0, 0] == 1);
        CHECK(t[0, 1] == 1);
        CHECK(t[1, 0] == 2);
        CHECK(t[1, 1] == 2);
    }

    SUBCASE("Constructor with array of blocks") {
        dynamic_tensor<int, error_checking::disabled> t_block1({2, 1});
        t_block1[0, 0] = 1;
        t_block1[1, 0] = 2;
        dynamic_tensor<int, error_checking::disabled> t_block2({2, 1});
        t_block2[0, 0] = 3;
        t_block2[1, 0] = 4;
        dynamic_tensor<int, error_checking::disabled> t_block3({2, 1});
        t_block3[0, 0] = 5;
        t_block3[1, 0] = 6;
        dynamic_tensor<int, error_checking::disabled> t({2, 3}, std::vector{t_block1, t_block2, t_block3});
        CHECK(t[0, 0] == 1);
        CHECK(t[1, 0] == 2);
        CHECK(t[0, 1] == 3);
        CHECK(t[1, 1] == 4);
        CHECK(t[0, 2] == 5);
        CHECK(t[1, 2] == 6);
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
        auto subview = t.subview({2,2}, {0,1});
        CHECK(subview[0, 0] == 2);
        CHECK(subview[1, 1] == 7);
    }

    SUBCASE("Modify through subview") {
        auto subview = t.subview({2,2}, {0,1});
        subview[0, 1] = 100;
        CHECK(t[0, 2] == 100);
    }

    SUBCASE("Assign from const tensor") {
        const dynamic_tensor<int, error_checking::disabled> const_tens = t;
        dynamic_tensor<int, error_checking::disabled> other_tens({3, 4});
        other_tens.view() = const_tens;
        CHECK(t[0, 0] == 1);
        CHECK(t[2, 3] == 12);
    }

    SUBCASE("Assign from const view") {
        const dynamic_tensor<int, error_checking::disabled> const_tens = t;
        dynamic_tensor<int, error_checking::disabled> other_tens({3, 4});
        other_tens.view() = const_tens.view();
        CHECK(t[0, 0] == 1);
        CHECK(t[2, 3] == 12);
    }
}

TEST_CASE("Dynamic Tensor Iteration") {
    dynamic_tensor<int, error_checking::disabled> t({2, 3});
    t[0, 0] = 1;
    t[1, 0] = 2;
    t[0, 1] = 3;
    t[1, 1] = 4;
    t[0, 2] = 5;
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
        dynamic_tensor<int, error_checking::enabled> t({2, 3});
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
        dynamic_tensor<int, error_checking::enabled> t({2, 3}, layout::row_major);
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

    SUBCASE("Fixed tensor reshape (column major)") {
        fixed_tensor<int, layout::column_major, error_checking::disabled, 2, 3> t;
        for (int i = 0; i < 6; ++i) {
            t[i / 3, i % 3] = i + 1;
        }

        auto reshaped_t = t.reshape<3, 2>();
        CHECK(reshaped_t.shape() == std::vector<std::size_t>{3, 2});
        CHECK(reshaped_t[0, 0] == 1);
        CHECK(reshaped_t[1, 1] == 3);
        CHECK(reshaped_t[2, 1] == 6);

        // t.reshape<2,2>(); // this line should not compile
    }

    SUBCASE("Fixed tensor reshape (row major)") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t;
        for (int i = 0; i < 6; ++i) {
            t[i / 3, i % 3] = i + 1;
        }

        auto reshaped_t = t.reshape<3, 2>();
        CHECK(reshaped_t.shape() == std::vector<std::size_t>{3, 2});
        CHECK(reshaped_t[0, 0] == 1);
        CHECK(reshaped_t[1, 1] == 4);
        CHECK(reshaped_t[2, 1] == 6);

        // t.reshape<2,2>(); // this line should not compile
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
        fixed_tensor<int, layout::row_major, error_checking::disabled, 1, 1> t{42};
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
        auto subview = view.subview<2, 3>(0,1);
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
        auto subview = view.subview({2,3}, {0,1}); 
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
        CHECK(values == std::vector<int>{1, 4, 2, 5, 3, 6});
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
        CHECK(values == std::vector<int>{1, 4, 2, 5, 3, 6});
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
        CHECK_THROWS_AS((t.subview<2, 2>(1,0)), std::out_of_range);
        CHECK_THROWS_AS((t.subview<2, 2>(0,2)), std::out_of_range);
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
        CHECK_THROWS_AS(t.subview({2,3}, {1,3}), std::out_of_range);
        CHECK_THROWS_AS(t.subview({2,1}, {0,3}), std::out_of_range);
    }
}

TEST_CASE("Tensor View Error Checking") {
    SUBCASE("Fixed tensor view with error checking") {
        fixed_tensor<int, layout::row_major, error_checking::enabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        auto view = t.view();

        CHECK_THROWS_AS(view.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(view.at(0, 3), std::out_of_range);
        CHECK_THROWS_AS((view.subview<2, 2>(1,0)), std::out_of_range);
    }

    SUBCASE("Dynamic tensor view with error checking") {
        dynamic_tensor<int, error_checking::enabled> t({2, 3});
        auto view = t.view();

        CHECK_THROWS_AS(view.at(2, 0), std::out_of_range);
        CHECK_THROWS_AS(view.at(0, 3), std::out_of_range);
        CHECK_THROWS_AS(view.subview({1,3}, {2,0}), std::out_of_range);
    }
}

TEST_CASE("Error Checking Disabled") {
    SUBCASE("Fixed tensor without error checking") {
        fixed_tensor<int, layout::row_major, error_checking::disabled, 2, 3> t{{1, 2, 3, 4, 5, 6}};
        // CHECK_NOTHROW(t.at(2, 0)); // This would be out of bounds, but no check is performed (can cause a crash)
        // CHECK_NOTHROW(t.at(0, 3)); // This would be out of bounds, but no check is performed (can cause a crash)
        CHECK_NOTHROW(t.subview<2, 2>(1,0)); // This would be invalid, but no check is performed
    }

    SUBCASE("Dynamic tensor without error checking") {
        dynamic_tensor<int, error_checking::disabled> t({2, 3});
        // CHECK_NOTHROW(t.at(2, 0)); // This would be out of bounds, but no check is performed (can cause a crash)
        // CHECK_NOTHROW(t.at(0, 3)); // This would be out of bounds, but no check is performed (can cause a crash)
        CHECK_NOTHROW(t.subview({1,0}, {2,3})); // This would be invalid, but no check is performed
    }
}

TEST_CASE("fixed_tensor static methods") {
    using tensor_type = squint::fixed_tensor<float, squint::layout::row_major, squint::error_checking::enabled, 2, 3>;

    SUBCASE("zeros") {
        auto t = tensor_type::zeros();
        CHECK(t.size() == 6);
        CHECK(t.at(0, 0) == 0.0F);
        CHECK(t.at(1, 2) == 0.0F);
    }

    SUBCASE("ones") {
        auto t = tensor_type::ones();
        CHECK(t.size() == 6);
        CHECK(t.at(0, 0) == 1.0F);
        CHECK(t.at(1, 2) == 1.0F);
    }

    SUBCASE("full") {
        auto t = tensor_type::full(3.14F);
        CHECK(t.size() == 6);
        CHECK(t.at(0, 0) == 3.14F);
        CHECK(t.at(1, 2) == 3.14F);
    }

    SUBCASE("arange") {
        auto t = tensor_type::arange(1.0F, 0.5F);
        CHECK(t.size() == 6);
        CHECK(t.at(0, 0) == 1.0F);
        CHECK(t.at(0, 1) == 1.5F);
        CHECK(t.at(0, 2) == 2.0F);
        CHECK(t.at(1, 0) == 2.5F);
        CHECK(t.at(1, 1) == 3.0F);
        CHECK(t.at(1, 2) == 3.5F);
    }

    SUBCASE("diag") {
        squint::fixed_tensor<float, squint::layout::row_major, squint::error_checking::enabled, 2> diag_vector(
            {1.0F, 2.0F});
        auto t = tensor_type::diag(diag_vector);
        CHECK(t.at(0, 0) == 1.0F);
        CHECK(t.at(1, 1) == 2.0F);
        CHECK(t.at(0, 1) == 0.0F);
        CHECK(t.at(1, 0) == 0.0F);
        CHECK(t.at(0, 2) == 0.0F);
        CHECK(t.at(1, 2) == 0.0F);
    }

    SUBCASE("diag with scalar") {
        auto t = tensor_type::diag(5.0F);
        CHECK(t.at(0, 0) == 5.0F);
        CHECK(t.at(1, 1) == 5.0F);
        CHECK(t.at(0, 1) == 0.0F);
        CHECK(t.at(1, 0) == 0.0F);
        CHECK(t.at(0, 2) == 0.0F);
        CHECK(t.at(1, 2) == 0.0F);
    }

    SUBCASE("random") {
        auto t = tensor_type::random(0.0F, 1.0F);
        CHECK(t.size() == 6);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(t.at(i, j) >= 0.0F);
                CHECK(t.at(i, j) <= 1.0F);
            }
        }
    }

    SUBCASE("I") {
        using square_tensor_type =
            squint::fixed_tensor<float, squint::layout::row_major, squint::error_checking::enabled, 3, 3>;
        auto t = square_tensor_type::eye();
        CHECK(t.at(0, 0) == 1.0F);
        CHECK(t.at(1, 1) == 1.0F);
        CHECK(t.at(2, 2) == 1.0F);
        CHECK(t.at(0, 1) == 0.0F);
        CHECK(t.at(1, 0) == 0.0F);
        CHECK(t.at(0, 2) == 0.0F);
        CHECK(t.at(2, 0) == 0.0F);
    }
}

TEST_CASE("dynamic_tensor static methods") {
    using tensor_type = squint::dynamic_tensor<float, squint::error_checking::enabled>;
    std::vector<std::size_t> shape = {2, 3};

    SUBCASE("zeros") {
        auto t = tensor_type::zeros(shape);
        CHECK(t.size() == 6);
        CHECK(t.at_impl({0, 0}) == 0.0F);
        CHECK(t.at_impl({1, 2}) == 0.0F);
    }

    SUBCASE("ones") {
        auto t = tensor_type::ones(shape);
        CHECK(t.size() == 6);
        CHECK(t.at_impl({0, 0}) == 1.0F);
        CHECK(t.at_impl({1, 2}) == 1.0F);
    }

    SUBCASE("full") {
        auto t = tensor_type::full(shape, 3.14F);
        CHECK(t.size() == 6);
        CHECK(t.at_impl({0, 0}) == 3.14F);
        CHECK(t.at_impl({1, 2}) == 3.14F);
    }

    SUBCASE("arange") {
        auto t = tensor_type::arange(shape, 1.0F, 0.5F);
        CHECK(t.size() == 6);
        CHECK(t.at_impl({0, 0}) == 1.0F);
        CHECK(t.at_impl({1, 0}) == 1.5F);
        CHECK(t.at_impl({0, 1}) == 2.0F);
        CHECK(t.at_impl({1, 1}) == 2.5F);
        CHECK(t.at_impl({0, 2}) == 3.0F);
        CHECK(t.at_impl({1, 2}) == 3.5F);
    }

    SUBCASE("diag") {
        tensor_type diag_vector({2}, 1.0F);
        diag_vector.at_impl({1}) = 2.0F;
        auto t = tensor_type::diag(diag_vector, shape);
        CHECK(t.at_impl({0, 0}) == 1.0F);
        CHECK(t.at_impl({1, 1}) == 2.0F);
        CHECK(t.at_impl({0, 1}) == 0.0F);
        CHECK(t.at_impl({1, 0}) == 0.0F);
        CHECK(t.at_impl({0, 2}) == 0.0F);
        CHECK(t.at_impl({1, 2}) == 0.0F);
    }

    SUBCASE("diag with scalar") {
        auto t = tensor_type::diag(5.0F, shape);
        CHECK(t.at_impl({0, 0}) == 5.0F);
        CHECK(t.at_impl({1, 1}) == 5.0F);
        CHECK(t.at_impl({0, 1}) == 0.0F);
        CHECK(t.at_impl({1, 0}) == 0.0F);
        CHECK(t.at_impl({0, 2}) == 0.0F);
        CHECK(t.at_impl({1, 2}) == 0.0F);
    }

    SUBCASE("random") {
        auto t = tensor_type::random(shape, 0.0F, 1.0F);
        CHECK(t.size() == 6);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(t.at_impl({i, j}) >= 0.0F);
                CHECK(t.at_impl({i, j}) <= 1.0F);
            }
        }
    }

    SUBCASE("I") {
        std::vector<std::size_t> square_shape = {3, 3};
        auto t = tensor_type::eye(square_shape);
        CHECK(t.at_impl({0, 0}) == 1.0F);
        CHECK(t.at_impl({1, 1}) == 1.0F);
        CHECK(t.at_impl({2, 2}) == 1.0F);
        CHECK(t.at_impl({0, 1}) == 0.0F);
        CHECK(t.at_impl({1, 0}) == 0.0F);
        CHECK(t.at_impl({0, 2}) == 0.0F);
        CHECK(t.at_impl({2, 0}) == 0.0F);
    }
}

TEST_CASE("fixed_tensor fill and flatten methods") {
    using tensor_type =
        squint::fixed_tensor<float, squint::layout::column_major, squint::error_checking::enabled, 2, 3>;

    SUBCASE("fill") {
        tensor_type t;
        t.fill(3.14F);
        CHECK(t.size() == 6);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(t.at(i, j) == 3.14F);
            }
        }
    }

    SUBCASE("flatten non-const") {
        tensor_type t = tensor_type::arange(1.0F, 1.0F);
        auto flattened = t.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at(i) == static_cast<float>(i + 1));
        }
        // Verify that modifications to flattened affect the original tensor
        flattened.at(0) = 10.0F;
        CHECK(t.at(0, 0) == 10.0F);
    }

    SUBCASE("flatten const") {
        const tensor_type t = tensor_type::arange(1.0F, 1.0F);
        auto flattened = t.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at(i) == static_cast<float>(i + 1));
        }
        // Verify that flattened is indeed const
        CHECK_FALSE(std::is_assignable<decltype(flattened.at(0)), float>::value);
    }

    SUBCASE("view flatten") {
        tensor_type t;
        t.fill(3.14F);
        auto view = t.view();
        auto flattened = view.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at(i) == 3.14F);
        }
        // Verify that modifications to flattened affect the original tensor
        flattened.at(0) = 10.0F;
        CHECK(t.at(0, 0) == 10.0F);
    }

    SUBCASE("view flatten const") {
        const tensor_type t(3.14F);
        const auto view = t.view();
        auto flattened = view.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at(i) == 3.14F);
        }
        // Verify that flattened is indeed const
        CHECK_FALSE(std::is_assignable<decltype(flattened.at(0)), float>::value);
    }
}

TEST_CASE("dynamic_tensor fill and flatten methods") {
    using tensor_type = squint::dynamic_tensor<float, squint::error_checking::enabled>;
    std::vector<std::size_t> shape = {2, 3};

    SUBCASE("fill") {
        tensor_type t(shape);
        t.fill(3.14F);
        CHECK(t.size() == 6);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(t.at_impl({i, j}) == 3.14F);
            }
        }
    }

    SUBCASE("flatten non-const") {
        tensor_type t = tensor_type::arange(shape, 1.0F, 1.0F);
        auto flattened = t.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at_impl({i}) == static_cast<float>(i + 1));
        }
        // Verify that modifications to flattened affect the original tensor
        flattened.at_impl({0}) = 10.0F;
        CHECK(t.at_impl({0, 0}) == 10.0F);
    }

    SUBCASE("flatten const") {
        const tensor_type t = tensor_type::arange(shape, 1.0F, 1.0F);
        auto flattened = t.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at_impl({i}) == static_cast<float>(i + 1));
        }
        // Verify that flattened is indeed const
        CHECK_FALSE(std::is_assignable<decltype(flattened.at_impl({0})), float>::value);
    }

    SUBCASE("view flatten") {
        tensor_type t(shape);
        t.fill(3.14F);
        auto view = t.view();
        auto flattened = view.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at_impl({i}) == 3.14F);
        }
        // Verify that modifications to flattened affect the original tensor
        flattened.at_impl({0}) = 10.0F;
        CHECK(t.at_impl({0, 0}) == 10.0F);
    }

    SUBCASE("view flatten const") {
        const tensor_type t(shape, 3.14F);
        const auto view = t.view();
        auto flattened = view.flatten();
        CHECK(flattened.size() == 6);
        CHECK(flattened.rank() == 1);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK(flattened.at_impl({i}) == 3.14F);
        }
        // Verify that flattened is indeed const
        CHECK_FALSE(std::is_assignable<decltype(flattened.at_impl({0})), float>::value);
    }
}

TEST_CASE("fixed_tensor rows() and cols() tests") {
    SUBCASE("2D tensor") {
        squint::fixed_tensor<int, squint::layout::row_major, squint::error_checking::disabled, 3, 4> tensor(
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

        SUBCASE("rows()") {
            std::vector<std::vector<int>> expected_rows = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
            int row_index = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1, 4});
                std::vector<int> row_values;
                for (const auto &value : row) {
                    row_values.push_back(value);
                }
                CHECK(row_values == expected_rows[row_index]);
                ++row_index;
            }
            CHECK(row_index == 3);
        }

        SUBCASE("cols()") {
            std::vector<std::vector<int>> expected_cols = {{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}};
            int col_index = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{3, 1});
                std::vector<int> col_values;
                for (const auto &value : col) {
                    col_values.push_back(value);
                }
                CHECK(col_values == expected_cols[col_index]);
                ++col_index;
            }
            CHECK(col_index == 4);
        }
    }

    SUBCASE("3D tensor") {
        squint::fixed_tensor<int, squint::layout::row_major, squint::error_checking::disabled, 2, 3, 4> tensor;
        for (std::size_t i = 0; i < 24; ++i) {
            tensor.at_impl({i / 12, (i % 12) / 4, i % 4}) = i + 1;
        }

        SUBCASE("rows()") {
            int row_index = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1, 3, 4});
                CHECK(row.size() == 12);
                for (std::size_t i = 0; i < 12; ++i) {
                    CHECK(row.at_impl({0, i / 4, i % 4}) == row_index * 12 + i + 1);
                }
                ++row_index;
            }
            CHECK(row_index == 2);
        }

        SUBCASE("cols()") {
            int col_index = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{2, 3, 1});
                CHECK(col.size() == 6);
                for (std::size_t i = 0; i < 6; ++i) {
                    CHECK(col.at_impl({i / 3, i % 3, 0}) == i * 4 + col_index + 1);
                }
                ++col_index;
            }
            CHECK(col_index == 4);
        }
    }

    SUBCASE("1D tensor") {
        squint::fixed_tensor<int, squint::layout::row_major, squint::error_checking::disabled, 5> tensor(
            {1, 2, 3, 4, 5});

        SUBCASE("rows()") {
            int row_count = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1});
                CHECK(row.size() == 1);
                CHECK(row[0] == row_count + 1);
                ++row_count;
            }
            CHECK(row_count == 5);
        }

        SUBCASE("cols()") {
            int col_count = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{5});
                CHECK(col.size() == 5);
                int i = 0;
                for (const auto &value : col) {
                    CHECK(value == ++i);
                }
                ++col_count;
            }
            CHECK(col_count == 1);
        }
    }
}

TEST_CASE("dynamic_tensor rows() and cols() tests") {
    SUBCASE("2D tensor") {
        squint::dynamic_tensor<int, squint::error_checking::disabled> tensor({3, 4}, squint::layout::row_major);
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.begin(), data.end(), tensor.data());

        SUBCASE("rows()") {
            std::vector<std::vector<int>> expected_rows = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
            int row_index = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1, 4});
                std::vector<int> row_values;
                for (const auto &value : row) {
                    row_values.push_back(value);
                }
                CHECK(row_values == expected_rows[row_index]);
                ++row_index;
            }
            CHECK(row_index == 3);
        }

        SUBCASE("cols()") {
            std::vector<std::vector<int>> expected_cols = {{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}};
            int col_index = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{3, 1});
                std::vector<int> col_values;
                for (const auto &value : col) {
                    col_values.push_back(value);
                }
                CHECK(col_values == expected_cols[col_index]);
                ++col_index;
            }
            CHECK(col_index == 4);
        }
    }

    SUBCASE("3D tensor") {
        squint::dynamic_tensor<int, squint::error_checking::disabled> tensor({2, 3, 4}, squint::layout::row_major);
        for (std::size_t i = 0; i < 24; ++i) {
            tensor.at_impl({i / 12, (i % 12) / 4, i % 4}) = i + 1;
        }

        SUBCASE("rows()") {
            int row_index = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1, 3, 4});
                CHECK(row.size() == 12);
                for (std::size_t i = 0; i < 12; ++i) {
                    CHECK(row.at_impl({0, i / 4, i % 4}) == row_index * 12 + i + 1);
                }
                ++row_index;
            }
            CHECK(row_index == 2);
        }

        SUBCASE("cols()") {
            int col_index = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{2, 3, 1});
                CHECK(col.size() == 6);
                for (std::size_t i = 0; i < 6; ++i) {
                    CHECK(col.at_impl({i / 3, i % 3, 0}) == i * 4 + col_index + 1);
                }
                ++col_index;
            }
            CHECK(col_index == 4);
        }
    }

    SUBCASE("1D tensor") {
        squint::dynamic_tensor<int, squint::error_checking::disabled> tensor({5}, squint::layout::row_major);
        std::vector<int> data = {1, 2, 3, 4, 5};
        std::copy(data.begin(), data.end(), tensor.data());

        SUBCASE("rows()") {
            int row_count = 0;
            for (const auto &row : tensor.rows()) {
                CHECK(row.shape() == std::vector<std::size_t>{1});
                CHECK(row.size() == 1);
                CHECK(row.at_impl({0}) == row_count + 1);
                ++row_count;
            }
            CHECK(row_count == 5);
        }

        SUBCASE("cols()") {
            int col_count = 0;
            for (const auto &col : tensor.cols()) {
                CHECK(col.shape() == std::vector<std::size_t>{5});
                CHECK(col.size() == 5);
                for (std::size_t i = 0; i < 5; ++i) {
                    CHECK(col.at_impl({i}) == i + 1);
                }
                ++col_count;
            }
            CHECK(col_count == 1);
        }
    }
}
