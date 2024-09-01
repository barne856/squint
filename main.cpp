#include <squint/squint.hpp>

using namespace squint;

auto main() -> int {
    tensor<float, dynamic, dynamic> a({2, 3}, std::vector<float>{1, 4, 2, 5, 3, 6});
    auto a_device = a.to_device();
    auto b_device = a_device * 2.0f;
    auto b_host = b_device.to_host();
    std::cout << "b_host: " << b_host << std::endl;
    auto permute_device = a_device.permute({1, 0});
    auto permute_host = permute_device.to_host();
    std::cout << "permute_host: " << permute_host << std::endl;
    return 0;
}