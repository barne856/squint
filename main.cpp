#include "squint/quantity.hpp"
#include <iostream>

using namespace squint::units;

int main() {
    auto l = length(1.0f);
    std::cout << l << std::endl;
    std::cout << l.as<feet_t>() << std::endl;
    auto s = squint::units::time(1.0f);
    velocity v = l / s;
    std::cout << v << std::endl;
    std::cout << v.as<miles_per_hour_t>() << std::endl;
    return 0;
}