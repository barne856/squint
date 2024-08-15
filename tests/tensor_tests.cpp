// NOLINTBEGIN
#include <utility>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/core/layout.hpp"
#include "squint/tensor.hpp"
#include <array>
#include <stdexcept>
#include <vector>

TEST_CASE("Tensor Construction and Basic Operations") {
    SUBCASE("Default construction") {
        squint::tensor<float, squint::shape<2, 3>> t;
        CHECK(t.size() == 6);
        CHECK(t.rank() == 2);
        CHECK(t.shape() == std::array<std::size_t, 2>{2, 3});
    }

    SUBCASE("Construction with initializer list (column-major)") {
        squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
        CHECK(t(0, 0) == 1);
        CHECK(t(1, 0) == 4);
        CHECK(t(0, 1) == 2);
        CHECK(t(1, 1) == 5);
        CHECK(t(0, 2) == 3);
        CHECK(t(1, 2) == 6);
    }

    SUBCASE("Construction with scalar value") {
        squint::tensor<float, squint::shape<2, 3>> t(42.0f);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(t(i, j) == 42.0f);
            }
        }
    }

    SUBCASE("Construction with std::array (column-major)") {
        std::array<float, 6> arr = {1, 4, 2, 5, 3, 6};
        squint::tensor<float, squint::shape<2, 3>> t(arr);
        CHECK(t(0, 0) == 1);
        CHECK(t(1, 0) == 4);
        CHECK(t(0, 1) == 2);
        CHECK(t(1, 1) == 5);
        CHECK(t(0, 2) == 3);
        CHECK(t(1, 2) == 6);
    }

    SUBCASE("Construction from other tensors") {
        squint::tensor<float, squint::shape<2, 2>> t1{1, 3, 2, 4};
        squint::tensor<float, squint::shape<2, 2>> t2{5, 7, 6, 8};
        squint::tensor<float, squint::shape<2, 2, 2>> t(t1, t2);
        CHECK(t(0, 0, 0) == 1);
        CHECK(t(1, 0, 0) == 3);
        CHECK(t(0, 1, 0) == 2);
        CHECK(t(1, 1, 0) == 4);
        CHECK(t(0, 0, 1) == 5);
        CHECK(t(1, 0, 1) == 7);
        CHECK(t(0, 1, 1) == 6);
        CHECK(t(1, 1, 1) == 8);
    }

    SUBCASE("Dynamic shape construction") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t({2, 3}, squint::layout::row_major);
        CHECK(t.size() == 6);
        CHECK(t.rank() == 2);
        CHECK(t.shape() == std::vector<std::size_t>{2, 3});
    }

    SUBCASE("Dynamic shape construction with initial values (row-major)") {
        std::vector<float> values = {1, 2, 3, 4, 5, 6};
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t({2, 3}, values,
                                                                                    squint::layout::row_major);
        CHECK(t(0, 0) == 1);
        CHECK(t(0, 1) == 2);
        CHECK(t(0, 2) == 3);
        CHECK(t(1, 0) == 4);
        CHECK(t(1, 1) == 5);
        CHECK(t(1, 2) == 6);
    }

    SUBCASE("Dynamic shape construction with initial values (column-major)") {
        std::vector<float> values = {1, 4, 2, 5, 3, 6};
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t({2, 3}, values,
                                                                                    squint::layout::column_major);
        CHECK(t(0, 0) == 1);
        CHECK(t(1, 0) == 4);
        CHECK(t(0, 1) == 2);
        CHECK(t(1, 1) == 5);
        CHECK(t(0, 2) == 3);
        CHECK(t(1, 2) == 6);
    }

    SUBCASE("Dynamic shape construction with scalar value") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t({2, 3}, 42.0f,
                                                                                    squint::layout::row_major);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(t(i, j) == 42.0f);
            }
        }
    }
}

TEST_CASE("Tensor Element Access") {
    squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};

    SUBCASE("Operator()") {
        CHECK(t(0, 0) == 1);
        CHECK(t(1, 0) == 4);
        CHECK(t(0, 1) == 2);
        CHECK(t(1, 1) == 5);
        CHECK(t(0, 2) == 3);
        CHECK(t(1, 2) == 6);
    }

    SUBCASE("Operator[] with std::array") {
        CHECK(t[{0, 0}] == 1);
        CHECK(t[{1, 0}] == 4);
        CHECK(t[{0, 1}] == 2);
        CHECK(t[{1, 1}] == 5);
        CHECK(t[{0, 2}] == 3);
        CHECK(t[{1, 2}] == 6);
    }

#ifndef _MSC_VER
    SUBCASE("Multidimensional operator[] (C++23)") {
        CHECK(t[0, 0] == 1);
        CHECK(t[1, 0] == 4);
        CHECK(t[0, 1] == 2);
        CHECK(t[1, 1] == 5);
        CHECK(t[0, 2] == 3);
        CHECK(t[1, 2] == 6);
    }
#endif

    SUBCASE("Const element access") {
        const auto &const_t = t;
        CHECK(const_t(0, 0) == 1);
        CHECK(const_t[{1, 0}] == 4);
#ifndef _MSC_VER
        CHECK(const_t[0, 2] == 3);
#endif
    }

    SUBCASE("Dynamic tensor indexing with std::vector") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> dt(
            {2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        CHECK(dt[std::vector<std::size_t>{0, 0}] == 1);
        CHECK(dt[std::vector<std::size_t>{1, 0}] == 4);
        CHECK(dt[std::vector<std::size_t>{0, 1}] == 2);
        CHECK(dt[std::vector<std::size_t>{1, 1}] == 5);
        CHECK(dt[std::vector<std::size_t>{0, 2}] == 3);
        CHECK(dt[std::vector<std::size_t>{1, 2}] == 6);
    }

    SUBCASE("non-const element access") {
        t(0, 0) = 42;
        CHECK(t(0, 0) == 42);
    }
}

TEST_CASE("Tensor Assignment") {
    SUBCASE("Fixed shape assignment") {
        squint::tensor<float, squint::shape<2, 3>> t1{1, 4, 2, 5, 3, 6};
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2 = t1;
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }

    SUBCASE("Dynamic shape assignment") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t1(
            {2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t2({2, 3});
        t2 = t1;
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }

    SUBCASE("Assignment with type conversion") {
        squint::tensor<int, squint::shape<2, 3>> t1{1, 4, 2, 5, 3, 6};
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2 = t1;
        CHECK(t2(0, 0) == 1.0f);
        CHECK(t2(1, 0) == 4.0f);
        CHECK(t2(0, 1) == 2.0f);
        CHECK(t2(1, 1) == 5.0f);
        CHECK(t2(0, 2) == 3.0f);
        CHECK(t2(1, 2) == 6.0f);
    }

    SUBCASE("Assigment to compatible shapes") {
        squint::tensor<float, squint::shape<2, 3, 1>> t1{1, 4, 2, 5, 3, 6};
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2 = t1;
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }

    SUBCASE("Assignment to tensor view") {
        squint::tensor<float, squint::shape<2, 3>> t1{1, 4, 2, 5, 3, 6};
        auto view = t1.view();
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2 = view;
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }

    SUBCASE("Assignment from tensor view") {
        squint::tensor<float, squint::shape<2, 3>> t1{1, 4, 2, 5, 3, 6};
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2 = t1.view();
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }

    SUBCASE("Assignment view to view") {
        squint::tensor<float, squint::shape<2, 3>> t1{1, 4, 2, 5, 3, 6};
        auto view1 = t1.view();
        auto view2 = view1.view();
        squint::tensor<float, squint::shape<2, 3>> t2;
        t2.view() = view2;
        CHECK(t2(0, 0) == 1);
        CHECK(t2(1, 0) == 4);
        CHECK(t2(0, 1) == 2);
        CHECK(t2(1, 1) == 5);
        CHECK(t2(0, 2) == 3);
        CHECK(t2(1, 2) == 6);
    }
}

TEST_CASE("Tensor Accessors") {
    squint::tensor<float, squint::shape<2, 3, 4>> t;

    SUBCASE("rank()") { CHECK(t.rank() == 3); }

    SUBCASE("shape()") {
        auto shape = t.shape();
        CHECK(shape == std::array<std::size_t, 3>{2, 3, 4});
    }

    SUBCASE("strides() for column-major layout") {
        auto strides = t.strides();
        CHECK(strides == std::array<std::size_t, 3>{1, 2, 6});
    }

    SUBCASE("size()") { CHECK(t.size() == 24); }

    SUBCASE("data()") { CHECK(t.data() != nullptr); }

    SUBCASE("const data()") {
        const auto &const_t = t;
        CHECK(const_t.data() != nullptr);
    }
}

TEST_CASE("Tensor Static Accessors") {
    using TensorType = squint::tensor<float, squint::shape<2, 3>>;

    SUBCASE("error_checking()") { CHECK(TensorType::error_checking() == squint::error_checking::disabled); }

    SUBCASE("ownership()") { CHECK(TensorType::ownership() == squint::ownership_type::owner); }

    SUBCASE("memory_space()") { CHECK(TensorType::memory_space() == squint::memory_space::host); }
}

TEST_CASE("Tensor Subview Operations") {
    SUBCASE("Fixed shape subview") {
        squint::tensor<float, squint::shape<3, 4, 5>> t;
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = static_cast<float>(i);
        }

        auto sub = t.template subview<2, 3>(1, 1, 1);
        CHECK(sub.rank() == 2);
        CHECK(sub.shape() == std::array<std::size_t, 2>{2, 3});
        CHECK(sub(0, 0) == t(1, 1, 1));
        CHECK(sub(1, 2) == t(2, 3, 1));
    }

    SUBCASE("Dynamic shape subview") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t({3, 4, 5});
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = static_cast<float>(i);
        }

        auto sub = t.subview({2, 3}, {1, 1, 1});
        CHECK(sub.rank() == 2);
        CHECK(sub.shape() == std::vector<std::size_t>{2, 3});
        CHECK(sub(0, 0) == t(1, 1, 1));
        CHECK(sub(1, 2) == t(2, 3, 1));
    }

    SUBCASE("Subview with step") {
        auto t = squint::tensor<float, squint::shape<4, 4>>::arange(1.0f, 1.0f);
        auto sub = t.subview<squint::shape<2, 2>, squint::seq<3, 3>>({0, 0});
        CHECK(sub(0, 0) == 1);
        CHECK(sub(1, 0) == 4);
        CHECK(sub(0, 1) == 13);
        CHECK(sub(1, 1) == 16);
    }
}

TEST_CASE("Tensor View Operations") {
    SUBCASE("Fixed shape view") {
        squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
        auto v = t.view();
        CHECK(v.data() == t.data());
        CHECK(v.shape() == t.shape());
        CHECK(v.strides() == t.strides());
    }

    SUBCASE("Dynamic shape view") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t(
            {2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        auto v = t.view();
        CHECK(v.data() == t.data());
        CHECK(v.shape() == t.shape());
        CHECK(v.strides() == t.strides());
    }

    SUBCASE("Diagonal view") {
        squint::tensor<float, squint::shape<3, 3>> t{1, 4, 7, 2, 5, 8, 3, 6, 9};
        auto diag = t.diag_view();
        CHECK(diag.rank() == 1);
        CHECK(diag.shape() == std::array<std::size_t, 1>{3});
        CHECK(diag(0) == 1);
        CHECK(diag(1) == 5);
        CHECK(diag(2) == 9);
    }

    SUBCASE("Dynamic shape diagonal view") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t(
            {3, 3}, std::vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9});
        auto diag = t.diag_view();
        CHECK(diag.rank() == 1);
        CHECK(diag.shape() == std::vector<std::size_t>{3});
        CHECK(diag(0) == 1);
        CHECK(diag(1) == 5);
        CHECK(diag(2) == 9);
    }
}

TEST_CASE("Tensor Static Creation Methods") {
    SUBCASE("zeros") {
        auto t = squint::tensor<float, squint::shape<2, 3>>::zeros();
        CHECK(t.size() == 6);
        for (size_t i = 0; i < t.size(); ++i) {
            CHECK(t.data()[i] == 0.0f);
        }
    }

    SUBCASE("ones") {
        auto t = squint::tensor<float, squint::shape<2, 3>>::ones();
        CHECK(t.size() == 6);
        for (size_t i = 0; i < t.size(); ++i) {
            CHECK(t.data()[i] == 1.0f);
        }
    }

    SUBCASE("full") {
        auto t = squint::tensor<float, squint::shape<2, 3>>::full(3.14f);
        CHECK(t.size() == 6);
        for (size_t i = 0; i < t.size(); ++i) {
            CHECK(t.data()[i] == 3.14f);
        }
    }

    SUBCASE("eye") {
        auto t = squint::tensor<float, squint::shape<3, 3>>::eye();
        CHECK(t(0, 0) == 1.0f);
        CHECK(t(1, 1) == 1.0f);
        CHECK(t(2, 2) == 1.0f);
        CHECK(t(0, 1) == 0.0f);
        CHECK(t(1, 0) == 0.0f);
    }

    SUBCASE("diag") {
        auto t = squint::tensor<float, squint::shape<3, 3>>::diag(2.0f);
        CHECK(t(0, 0) == 2.0f);
        CHECK(t(1, 1) == 2.0f);
        CHECK(t(2, 2) == 2.0f);
        CHECK(t(0, 1) == 0.0f);
        CHECK(t(1, 0) == 0.0f);
    }

    SUBCASE("arange") {
        auto t = squint::tensor<float, squint::shape<2, 3>>::arange(1.0f, 0.5f);
        CHECK(t(0, 0) == 1.0f);
        CHECK(t(1, 0) == 1.5f);
        CHECK(t(0, 1) == 2.0f);
        CHECK(t(1, 1) == 2.5f);
        CHECK(t(0, 2) == 3.0f);
        CHECK(t(1, 2) == 3.5f);
    }
}

TEST_CASE("Tensor Shape Manipulation") {
    SUBCASE("Fixed shape reshape") {
        squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
        auto reshaped = t.template reshape<3, 2>();
        CHECK(reshaped.shape() == std::array<std::size_t, 2>{3, 2});
        CHECK(reshaped(0, 0) == 1);
        CHECK(reshaped(1, 0) == 4);
        CHECK(reshaped(2, 0) == 2);
        CHECK(reshaped(0, 1) == 5);
        CHECK(reshaped(1, 1) == 3);
        CHECK(reshaped(2, 1) == 6);
    }

    SUBCASE("Dynamic shape reshape") {
        squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>> t(
            {2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
        t.reshape({3, 2});
        CHECK(t.shape() == std::vector<std::size_t>{3, 2});
        CHECK(t(0, 0) == 1);
        CHECK(t(1, 0) == 4);
        CHECK(t(2, 0) == 2);
        CHECK(t(0, 1) == 5);
        CHECK(t(1, 1) == 3);
        CHECK(t(2, 1) == 6);
    }

    SUBCASE("Flatten") {
        squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
        auto flattened = t.flatten();
        CHECK(flattened.rank() == 1);
        CHECK(flattened.shape() == std::array<std::size_t, 1>{6});
        for (size_t i = 0; i < 6; ++i) {
            CHECK(flattened(i) == t.data()[i]);
        }
    }

    SUBCASE("Permute") {
        squint::tensor<float, squint::shape<2, 3, 4>> t;
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = static_cast<float>(i);
        }
        auto permuted = t.template permute<1, 2, 0>();
        CHECK(permuted.shape() == std::array<std::size_t, 3>{3, 4, 2});
        CHECK(permuted(0, 0, 0) == t(0, 0, 0));
        CHECK(permuted(1, 2, 1) == t(1, 1, 2));
    }

    SUBCASE("Transpose") {
        squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
        auto transposed = t.transpose();
        CHECK(transposed.shape() == std::array<std::size_t, 2>{3, 2});
        CHECK(transposed(0, 0) == 1);
        CHECK(transposed(0, 1) == 4);
        CHECK(transposed(1, 0) == 2);
        CHECK(transposed(1, 1) == 5);
        CHECK(transposed(2, 0) == 3);
        CHECK(transposed(2, 1) == 6);
    }
}

TEST_CASE("Tensor Iteration Methods") {
    squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};

    SUBCASE("begin() and end()") {
        std::vector<float> values;
        for (auto it = t.begin(); it != t.end(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<float>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("cbegin() and cend()") {
        std::vector<float> values;
        for (auto it = t.cbegin(); it != t.cend(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<float>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("rows()") {
        std::vector<std::vector<float>> row_values;
        for (auto row : t.rows()) {
            std::vector<float> row_data{};
            for (const auto &val : row) {
                row_data.push_back(val);
            }
            row_values.push_back(row_data);
        }
        CHECK(row_values == std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
    }

    SUBCASE("cols()") {
        std::vector<std::vector<float>> col_values;
        for (auto col : t.cols()) {
            std::vector<float> col_data(col.size());
            std::copy(col.begin(), col.end(), col_data.begin());
            col_values.push_back(col_data);
        }
        CHECK(col_values == std::vector<std::vector<float>>{{1, 4}, {2, 5}, {3, 6}});
    }

    SUBCASE("subviews()") {
        squint::tensor<float, squint::shape<4, 4>> t{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<float> subview_sums;
        for (auto subview : t.template subviews<squint::shape<2, 2>>()) {
            subview_sums.push_back(std::accumulate(subview.begin(), subview.end(), 0.0f));
        }
        CHECK(subview_sums == std::vector<float>{14, 22, 46, 54});
    }
}

TEST_CASE("Error Checking") {
    using ErrorTensor =
        squint::tensor<float, squint::shape<2, 3>, squint::shape<3, 1>, squint::error_checking::enabled>;
    ErrorTensor t{1, 4, 2, 5, 3, 6};

    SUBCASE("Out of bounds access") {
        CHECK_THROWS_AS(t(2, 0), std::out_of_range);
        CHECK_THROWS_AS(t(0, 3), std::out_of_range);
    }
}

TEST_CASE("Memory Space") {
    using DeviceTensor =
        squint::tensor<float, squint::shape<2, 3>, squint::shape<3, 1>, squint::error_checking::disabled,
                       squint::ownership_type::owner, squint::memory_space::device>;

    SUBCASE("Device memory space") {
        DeviceTensor t;
        CHECK(t.memory_space() == squint::memory_space::device);
    }
}

TEST_CASE("Random Tensor Creation") {
    auto t = squint::tensor<float, squint::shape<2, 3>>::random(0.0f, 1.0f);

    SUBCASE("Values within range") {
        for (float val : t) {
            CHECK(val >= 0.0f);
            CHECK(val <= 1.0f);
        }
    }
}

TEST_CASE("Const Correctness") {
    const squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};

    SUBCASE("Const element access") {
        CHECK(t(0, 0) == 1);
        CHECK(t[{1, 2}] == 6);
    }

    SUBCASE("Const iteration") {
        std::vector<float> values;
        for (auto it = t.cbegin(); it != t.cend(); ++it) {
            values.push_back(*it);
        }
        CHECK(values == std::vector<float>{1, 4, 2, 5, 3, 6});
    }

    SUBCASE("Const views") {
        auto view = t.view();
        CHECK(view(0, 0) == 1);
        CHECK(view(1, 2) == 6);
    }
}

TEST_CASE("Edge Cases") {
    SUBCASE("Zero-dimensional tensor") {
        squint::tensor<float, squint::shape<>> t{42};
        CHECK(t.rank() == 0);
        CHECK(t.size() == 1);
        CHECK(t() == 42);
    }

    SUBCASE("Empty tensor") {
        squint::tensor<float, squint::shape<0>> t;
        CHECK(t.rank() == 1);
        CHECK(t.size() == 0);
    }
}

TEST_CASE("Dynamic Tensor Operations") {
    using DynamicTensor = squint::tensor<float, std::vector<std::size_t>, std::vector<std::size_t>>;

    SUBCASE("Dynamic tensor construction and basic operations") {
        DynamicTensor t({2, 3, 4});
        CHECK(t.rank() == 3);
        CHECK(t.shape() == std::vector<std::size_t>{2, 3, 4});
        CHECK(t.size() == 24);

        // Fill with incremental values
        float val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            *it = val;
        }

        // Check values
        val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            CHECK(*it == val);
        }
    }

    SUBCASE("Dynamic tensor reshape") {
        DynamicTensor t({2, 3, 4});
        float val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            *it = val;
        }

        t.reshape({4, 6});
        CHECK(t.rank() == 2);
        CHECK(t.shape() == std::vector<std::size_t>{4, 6});
        CHECK(t.size() == 24);

        // Check values after reshape
        val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            CHECK(*it == val);
        }
    }

    SUBCASE("Dynamic tensor subview") {
        DynamicTensor t({4, 5, 6});
        float val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            *it = val;
        }

        auto sub = t.subview({2, 3, 2}, {1, 1, 2});
        CHECK(sub.rank() == 3);
        CHECK(sub.shape() == std::vector<std::size_t>{2, 3, 2});

        // Check subview values
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 2; ++k) {
                    CHECK(sub(i, j, k) == t(i + 1, j + 1, k + 2));
                }
            }
        }
    }

    SUBCASE("Dynamic tensor permute") {
        DynamicTensor t({2, 3, 4});
        float val = 0;
        for (auto it = t.begin(); it != t.end(); ++it, val += 1.0f) {
            *it = val;
        }

        auto permuted = t.permute({2, 0, 1});
        CHECK(permuted.shape() == std::vector<std::size_t>{4, 2, 3});

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    CHECK(permuted(k, i, j) == t(i, j, k));
                }
            }
        }
    }

    SUBCASE("Dynamic tensor static creation methods") {
        auto zeros = DynamicTensor::zeros({2, 3, 4});
        CHECK(zeros.shape() == std::vector<std::size_t>{2, 3, 4});
        for (float val : zeros) {
            CHECK(val == 0.0f);
        }

        auto ones = DynamicTensor::ones({3, 4});
        CHECK(ones.shape() == std::vector<std::size_t>{3, 4});
        for (float val : ones) {
            CHECK(val == 1.0f);
        }

        auto full = DynamicTensor::full(3.14f, {2, 2});
        CHECK(full.shape() == std::vector<std::size_t>{2, 2});
        for (float val : full) {
            CHECK(val == 3.14f);
        }

        auto eye = DynamicTensor::eye({3, 3});
        CHECK(eye.shape() == std::vector<std::size_t>{3, 3});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (i == j) {
                    CHECK(eye(i, j) == 1.0f);
                } else {
                    CHECK(eye(i, j) == 0.0f);
                }
            }
        }

        auto diag = DynamicTensor::diag(2.0f, {3, 3});
        CHECK(diag.shape() == std::vector<std::size_t>{3, 3});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (i == j) {
                    CHECK(diag(i, j) == 2.0f);
                } else {
                    CHECK(diag(i, j) == 0.0f);
                }
            }
        }

        auto random = DynamicTensor::random(0.0f, 1.0f, {2, 3});
        CHECK(random.shape() == std::vector<std::size_t>{2, 3});
        for (float val : random) {
            CHECK(val >= 0.0f);
            CHECK(val <= 1.0f);
        }

        auto arange = DynamicTensor::arange(1.0f, 0.5f, {2, 3});
        CHECK(arange(0, 0) == 1.0f);
        CHECK(arange(1, 0) == 1.5f);
        CHECK(arange(0, 1) == 2.0f);
        CHECK(arange(1, 1) == 2.5f);
        CHECK(arange(0, 2) == 3.0f);
        CHECK(arange(1, 2) == 3.5f);
    }

    SUBCASE("Dynamic tensor iteration methods") {
        DynamicTensor t({2, 3});
        t(0, 0) = 0;
        t(0, 1) = 1;
        t(0, 2) = 2;
        t(1, 0) = 3;
        t(1, 1) = 4;
        t(1, 2) = 5;

        SUBCASE("rows()") {
            std::vector<std::vector<float>> row_values;
            for (auto row : t.rows()) {
                std::vector<float> row_data{};
                for (const auto &val : row) {
                    row_data.push_back(val);
                }
                row_values.push_back(row_data);
            }
            CHECK(row_values == std::vector<std::vector<float>>{{0, 1, 2}, {3, 4, 5}});
        }

        SUBCASE("cols()") {
            std::vector<std::vector<float>> col_values;
            for (auto col : t.cols()) {
                std::vector<float> col_data(col.size());
                std::copy(col.begin(), col.end(), col_data.begin());
                col_values.push_back(col_data);
            }
            CHECK(col_values == std::vector<std::vector<float>>{{0, 3}, {1, 4}, {2, 5}});
        }

        SUBCASE("subviews()") {
            DynamicTensor t({4, 4}, std::vector<float>{
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                    });

            std::vector<float> subview_sums;
            for (auto subview : t.subviews({2, 2})) {
                subview_sums.push_back(std::accumulate(subview.begin(), subview.end(), 0.0f));
            }
            CHECK(subview_sums == std::vector<float>{14, 22, 46, 54});
        }
    }
}

// NOLINTEND