#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;

    auto d = fixed_tensor<double, layout::column_major, error_checking::disabled, 4>::arange(4);
    auto eye = fixed_tensor<double, layout::column_major, error_checking::disabled, 4, 4>::diag(d);
    auto rand = fixed_tensor<double, layout::column_major, error_checking::disabled, 4, 4>::random();
    std::cout << eye << std::endl;
    std::cout << rand << std::endl;

    return 0;
}