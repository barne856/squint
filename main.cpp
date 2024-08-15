#include "squint/core/concepts.hpp"
#include "squint/core/layout.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <utility>
#include <vector>

using namespace squint;

auto main() -> int {
    squint::tensor<float, squint::shape<2, 3>> t{1, 4, 2, 5, 3, 6};
    std::cout << t << std::endl;
    for (const auto& row : t.rows()) {
        //std::cout << row << std::endl;
        std::cout << tens<1,3>(row) << std::endl;
    }
    //for (auto col : t.cols()) {
    //    std::cout << col << std::endl;
    //}
}