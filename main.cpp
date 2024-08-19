#include "squint/squint.hpp"

using namespace squint;

auto main() -> int {
    // Create two 4D tensors
    auto A = tens({3, 2, 2, 3});
    auto B = tens({3, 2, 2, 3});

    // Fill tensors with values
    auto A_flat = A.flatten();
    auto B_flat = B.flatten();
    for (size_t i = 0; i < 36; ++i) {
        A_flat[i] = static_cast<float>(i);
        B_flat[i] = static_cast<float>(i + 36);
    }

    std::cout << "Tensor A:" << std::endl << A << std::endl;
    std::cout << "Tensor B:" << std::endl << B << std::endl;

    // Perform contraction
    auto C = squint::contract(A, B, {{1, 2}, {2, 1}});

    std::cout << "Result:" << std::endl << C << std::endl;

    return 0;
}