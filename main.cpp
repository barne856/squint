#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

int main() {
    tens<2, 2> A{1.457347, 2.34573456737354673, 3.14, 4};

    std::cout << A << std::endl;
}