#include "squint/tensor/tensor_ops.hpp"
#include "squint/tensor/tensor_types.hpp"
#include "squint/util/array_utils.hpp"
#include <iostream>
#include <utility>

using namespace squint;

int main() {
    tens<2> B{1, 2};
    std::cout << B << std::endl;
}