// NOLINTBEGIN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "squint/core/concepts.hpp"
#include "squint/tensor/tensor.hpp"

using namespace squint;

TEST_CASE("Fixed tensors can be iterated over") {
    static_assert(tensorial<tensor<int, std::index_sequence<2, 3>>>);
    tensor<int, std::index_sequence<2, 3>> tensor{1, 2, 3, 4, 5, 6};
    int i = 1;
    for (auto &elem : tensor) {
        CHECK(elem == i++);
    }
}

// NOLINTEND