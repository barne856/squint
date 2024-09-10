#include <chrono>
#include <iostream>
#include <squint/squint.hpp>
#include <vector>

using namespace squint;

auto main() -> int {
    // create N random vec3s
    constexpr size_t N = 10000;
    std::vector<vec3> a_vecs;
    a_vecs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        a_vecs.push_back(vec3::random(0, 1));
    }
    std::vector<vec3> b_vecs;
    b_vecs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        b_vecs.push_back(vec3::random(0, 1));
    }
    std::vector<vec3> c_vecs;
    c_vecs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        c_vecs.push_back(vec3::random(0, 1));
    }
    // start clock
    auto start = std::chrono::high_resolution_clock::now();
    // perform N cross products
    for (size_t i = 0; i < N; i++) {
        cross(a_vecs[i], b_vecs[i], c_vecs[i]);
    }
    // stop clock
    auto stop = std::chrono::high_resolution_clock::now();
    // print seconds elapsed
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    // sum all results to force compiler to not optimize anything out
    vec3 sum{};
    for (size_t i = 0; i < N; ++i) {
        sum += c_vecs[i];
    }
    std::cout << sum << std::endl;

    return 0;
}