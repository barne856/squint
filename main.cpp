#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

int main() {
    tens<2, 2> A{1.457347, 2.34573456737354673, 3.14, 4};

    std::cout << A << std::endl;

    tens<2,3> B{1, 2, 3, 4, 5, 6};
    std::cout << B << std::endl;
    std::cout << B.transpose() << std::endl;
}