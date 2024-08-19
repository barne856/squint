#include "squint/squint.hpp"

using namespace squint;

auto main() -> int {
    auto a = ndarr<5, 4, 6>::arange(1, 1);
    ndarr<5, 24> b = a.reshape<5, 24>();
    ndarr<24, 5> b_t = b.transpose();
    std::cout << b_t.reshape<4, 30>() << std::endl;
}