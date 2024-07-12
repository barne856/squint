#include "squint/dimension.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <vector>

using namespace squint;
using namespace squint::units;

int main() {
    const auto A = tens::arange({4, 4}, 1);
    std::cout << A << std::endl;
    for (const auto &view : A.subviews({2, 2})) {
        std::cout << view << std::endl;
    }

    return 0;
}