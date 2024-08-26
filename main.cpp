#include <squint/squint.hpp>

using namespace squint;

auto main() -> int {
    auto A = tens::arange(1, 1, {2, 2, 2});
    auto P = tens(A.permute({0, 2, 1}));
    auto R = P.reshape({4, 2});
    std::cout << P << std::endl;
    std::cout << R << std::endl;
    auto B = tens::arange(1, 1, {2, 2, 2});
    std::cout << "A" << std::endl;
    std::cout << A << std::endl;
    std::cout << "B" << std::endl;
    std::cout << B << std::endl;
    std::vector<std::pair<size_t, size_t>> contraction_pairs = {{1, 0}};
    auto result = contract(A, B, contraction_pairs);
    std::cout << "Result" << std::endl;
    std::cout << result << std::endl;
    return 0;
}