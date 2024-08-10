// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/core/concepts.hpp"
#include "squint/tensor/tensor_types.hpp"

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

     // flat iterator
     int i = 0;
     std::array<int, 6> expected{1, 3, 5, 2, 4, 6};
     for (auto &elem : tensor) {
         CHECK(elem == expected[i++]);
     }
     // subview iterator
     i = 0;
     for (auto subview : tensor.subviews<2, 1>()) {
         CHECK(subview.ownership_type() == ownership_type::reference);
         CHECK(subview.size() == 2);
         CHECK(subview.strides()[0] == 1);
         CHECK(subview.strides()[1] == 2);
         CHECK(subview[{0, 0}] == 2 * i + 1);
         CHECK(subview[{1, 0}] == 2 * i + 2);
         i++;
     }
}

// NOLINTEND