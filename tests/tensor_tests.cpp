// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/core/concepts.hpp"
#include "squint/tensor/tensor.hpp"

using namespace squint;

TEST_CASE("Fixed tensors can be iterated over") {
    static_assert(tensorial<tensor<int, std::index_sequence<2, 3>>>);
    tens<2, 3> tensor{1, 2, 3, 4, 5, 6};

    // size should be 6
    CHECK(tensor.size() == 6);
    // class should be 24 bytes
    CHECK(sizeof(tens<2, 3>) == 24);
    // check strides
    CHECK(tensor.strides()[0] == 1);
    CHECK(tensor.strides()[1] == 2);

    // check subscript operator
    CHECK(tensor[{0, 0}] == 1);
    CHECK(tensor[{1, 0}] == 2);
    CHECK(tensor[{0, 1}] == 3);
    CHECK(tensor[{1, 1}] == 4);
    CHECK(tensor[{0, 2}] == 5);
    CHECK(tensor[{1, 2}] == 6);
    


    // int i = 1;
    // for (auto &elem : tensor) {
    //     CHECK(elem == i++);
    // }
}

// NOLINTEND