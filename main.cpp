#include <chrono>
#include <iostream>
#include <squint/squint.hpp>

using namespace squint;
using namespace std::chrono;

auto main() -> int {
    auto a = tensor<float, dynamic, dynamic>::arange(1, 1, {10000, 10000});
    auto b = tensor<float, dynamic, dynamic>::arange(1, 1, {10000, 10000});

    // Time host multiplication
    auto start_host = high_resolution_clock::now();
    auto c_host = a * b;
    auto end_host = high_resolution_clock::now();
    auto duration_host = duration_cast<milliseconds>(end_host - start_host);

    // Time transfer to device
    auto start_transfer_to_device = high_resolution_clock::now();
    auto a_device = a.to_device();
    auto b_device = b.to_device();
    auto end_transfer_to_device = high_resolution_clock::now();
    auto duration_transfer_to_device = duration_cast<milliseconds>(end_transfer_to_device - start_transfer_to_device);

    // Time device multiplication
    auto start_device = high_resolution_clock::now();
    auto c_device = a_device * b_device;
    auto end_device = high_resolution_clock::now();
    auto duration_device = duration_cast<milliseconds>(end_device - start_device);

    // Time transfer from device
    auto start_transfer_from_device = high_resolution_clock::now();
    auto c_host_from_device = c_device.to_host();
    auto end_transfer_from_device = high_resolution_clock::now();
    auto duration_transfer_from_device =
        duration_cast<milliseconds>(end_transfer_from_device - start_transfer_from_device);

    // Print results
    std::cout << "Host multiplication time: " << duration_host.count() << " ms\n";
    std::cout << "Transfer to device time: " << duration_transfer_to_device.count() << " ms\n";
    std::cout << "Device multiplication time: " << duration_device.count() << " ms\n";
    std::cout << "Transfer from device time: " << duration_transfer_from_device.count() << " ms\n";

    // Calculate and print total device time
    auto total_device_time = duration_transfer_to_device + duration_device + duration_transfer_from_device;
    std::cout << "Total device time (including transfers): " << total_device_time.count() << " ms\n";

    return 0;
}