#include "squint/quantity.hpp"
#include <iostream>

using namespace squint::units;

int main() {
    auto l = length(1.0F) * 2;
    float test = float(l);
    // float test = l; // compile error
    std::cout << l << std::endl;
    std::cout << l.as<feet_t>() << std::endl;
    auto s = squint::units::time(1.0F);
    velocity v = l / s;
    std::cout << v << std::endl;
    std::cout << v.as<miles_per_hour_t>() << std::endl;

    auto a = squint::units::acceleration_t<float, squint::error_checking_enabled>(1.0F);
    std::cout << a / 0 << std::endl;
    return 0;
}