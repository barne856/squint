#include "squint/dimension.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <vector>

using namespace squint;
using namespace squint::units;

int main() {
    auto A = mat4_t<units::dimensionless>::random();
    std::cout << A << std::endl;
    auto b = vec4::random();
    std::cout << b << std::endl;
    A.solve(b);
    std::cout << b << std::endl;
    return 0;
}