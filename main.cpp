#include "squint/core/concepts.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>
#include <utility>

using namespace squint;

auto main() -> int {
    using test_t = std::common_type_t<float, double>;
    tens<2, 2> A{1, 2, 3, 4};
    auto B = tensor<float, shape<2, 2>>::arange(float(1), float(1));
    static_assert(fixed_contiguous_tensor<decltype(B.subview<shape<2>, seq<1>>({0, 0}))>);
    std::cout << A << std::endl;
    std::cout << B.subview<2>(0, 0) << std::endl;
    std::cout << A * B.subview<2>(0, 0) << std::endl;
}