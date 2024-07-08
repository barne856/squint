#include "squint/fixed_tensor.hpp"
#include "squint/tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;

    fixed_tensor<int, layout::row_major, error_checking::enabled, 2, 3> a{1};
    for (auto v : a.subviews<2, 1>()) {
        auto u = v.reshape<2>();
        u[0] = 2;
        u[1] = 2;
        v = u;
    }
    std::cout << a << std::endl;

    return 0;
}