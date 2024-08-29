#include <iostream>
#include <squint/squint.hpp>

using namespace squint;

auto main() -> int {
    auto A = tensor<float, shape<2, 2, 2>>::arange(1, 1);
    auto B = tensor<float, shape<2, 2, 2>>::arange(1, 1);
    auto result = contract(A, B, std::index_sequence<1>{}, std::index_sequence<0>{});
    std::cout << result << std::endl;
    return 0;
}