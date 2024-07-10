#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <chrono>
#include <iostream>

using namespace squint;
using namespace squint::units;

void func(length t) { std::cout << "float: " << t << std::endl; }

int main() {

    mat3 a({1, 2, 3, 4, 5, 6, 7, 8, 9});
    vec3 b({1, 2, 3});
    // a.solve(b);
    return 0;
}