#include <iostream>
#include <squint/squint.hpp>

using namespace squint;

auto main() -> int {
    auto A = tensor<float, shape<2, 2, 2>>::arange(1, 1);
    auto B = tensor<float, shape<2, 2, 2>>::arange(1, 1);
    constexpr auto contraction_pairs = make_contraction_pairs(std::index_sequence<1>{}, // A indices
                                                              std::index_sequence<0>{}  // B indices
    );
    // std::cout << contraction_pairs().size() << std::endl;
    // std::cout << std::get<0>(decltype(contraction_pairs)::value[0]) << std::endl;
    // std::cout << std::get<1>(decltype(contraction_pairs)::value[0]) << std::endl;
    // for (size_t i = 0; contraction_pairs().size(); ++i) {
    //     std::cout << contraction_pairs()[i].first << " ";
    //     std::cout << contraction_pairs()[i].second << " ";
    // }
    auto result = contract(A, B, contraction_pairs);
    std::cout << result << std::endl;
    return 0;
}