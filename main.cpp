#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;
    using namespace squint::units;

    auto eye = mat4<length>::I();

    std::cout << eye << std::endl;

    return 0;
}