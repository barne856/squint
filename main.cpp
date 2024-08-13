#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

auto main() -> int {
    using test_t = decltype(std::declval<float>() * std::declval<length>());
    tens<4, 4> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto B = tensor<pure, shape<4, 4>>::arange(1, 1);

    std::cout << A * length(6) << std::endl;
}