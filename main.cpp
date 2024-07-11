#include "squint/dimension.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <vector>

using namespace squint;
using namespace squint::units;

int main() {
    auto A = mat4::random();
    for (const auto &block : A.subviews<2, 2>()) {
        for(const auto& elem : block) {
            std::cout << elem << std::endl;
        }
    }
    std::cout << A << std::endl;
    return 0;
}