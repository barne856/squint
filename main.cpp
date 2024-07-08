#include "squint/fixed_tensor.hpp"
#include "squint/tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;
    fixed_tensor<int, layout::row_major, 3, 4> t{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::cout << t << std::endl;
    std::vector<std::vector<int>> subview_values;
    for (const auto &subview : t.subviews<2, 2>()) {
        std::vector<int> values;
        for (const auto &value : subview) {
            values.push_back(value);
        }
        subview_values.push_back(values);
    }
    for (const auto &values : subview_values) {
        for (const auto &value : values) {
            std::cout << std::setw(2) << value << " ";
        }
        std::cout << std::endl;
    }
    // subview_values == std::vector<std::vector<int>>{{1, 2, 5, 6}, {3, 4, 7, 8}, {9, 10, 11, 12}};

    return 0;
}