#include <chrono>
// #include <immintrin.h>
#include <iostream>
#include <squint/squint.hpp>
#include <stdio.h>
#include <vector>

using namespace squint;

// void invert_matrix_4x4_avx(const float *A, float *B) {
//     __m256 row0 = _mm256_loadu_ps(&A[0]);
//     __m256 row1 = _mm256_loadu_ps(&A[4]);
//     __m256 row2 = _mm256_loadu_ps(&A[8]);
//     __m256 row3 = _mm256_loadu_ps(&A[12]);
// 
//     __m256 tmp1 = _mm256_shuffle_ps(row0, row1, 0x44);
//     __m256 tmp2 = _mm256_shuffle_ps(row2, row3, 0x44);
//     __m256 tmp3 = _mm256_shuffle_ps(row0, row1, 0xEE);
//     __m256 tmp4 = _mm256_shuffle_ps(row2, row3, 0xEE);
// 
//     __m256 row0_2 = _mm256_shuffle_ps(tmp1, tmp2, 0x88);
//     __m256 row1_2 = _mm256_shuffle_ps(tmp1, tmp2, 0xDD);
//     __m256 row2_2 = _mm256_shuffle_ps(tmp3, tmp4, 0x88);
//     __m256 row3_2 = _mm256_shuffle_ps(tmp3, tmp4, 0xDD);
// 
//     tmp1 = _mm256_mul_ps(row2_2, row3_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
// 
//     __m256 minor0 = _mm256_mul_ps(row1_2, tmp1);
//     __m256 minor1 = _mm256_mul_ps(row0_2, tmp1);
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor0 = _mm256_sub_ps(_mm256_mul_ps(row1_2, tmp1), minor0);
//     minor1 = _mm256_sub_ps(_mm256_mul_ps(row0_2, tmp1), minor1);
//     minor1 = _mm256_permute_ps(minor1, 0xB1);
// 
//     tmp1 = _mm256_mul_ps(row1_2, row2_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
//     minor0 = _mm256_add_ps(_mm256_mul_ps(row3_2, tmp1), minor0);
//     __m256 minor3 = _mm256_mul_ps(row0_2, tmp1);
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor0 = _mm256_sub_ps(minor0, _mm256_mul_ps(row3_2, tmp1));
//     minor3 = _mm256_sub_ps(_mm256_mul_ps(row0_2, tmp1), minor3);
//     minor3 = _mm256_permute_ps(minor3, 0xB1);
// 
//     tmp1 = _mm256_mul_ps(_mm256_permute_ps(row1_2, 0x4E), row3_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
//     __m256 row2_2_perm = _mm256_permute_ps(row2_2, 0x4E);
//     minor0 = _mm256_add_ps(_mm256_mul_ps(row2_2_perm, tmp1), minor0);
//     __m256 minor2 = _mm256_mul_ps(row0_2, tmp1);
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor0 = _mm256_sub_ps(minor0, _mm256_mul_ps(row2_2_perm, tmp1));
//     minor2 = _mm256_sub_ps(_mm256_mul_ps(row0_2, tmp1), minor2);
//     minor2 = _mm256_permute_ps(minor2, 0xB1);
// 
//     tmp1 = _mm256_mul_ps(row0_2, row1_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
//     minor2 = _mm256_add_ps(_mm256_mul_ps(row3_2, tmp1), minor2);
//     minor3 = _mm256_sub_ps(_mm256_mul_ps(row2_2, tmp1), minor3);
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor2 = _mm256_sub_ps(_mm256_mul_ps(row3_2, tmp1), minor2);
//     minor3 = _mm256_sub_ps(minor3, _mm256_mul_ps(row2_2, tmp1));
// 
//     tmp1 = _mm256_mul_ps(row0_2, row3_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
//     minor1 = _mm256_sub_ps(minor1, _mm256_mul_ps(row2_2, tmp1));
//     minor2 = _mm256_add_ps(_mm256_mul_ps(row1_2, tmp1), minor2);
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor1 = _mm256_add_ps(_mm256_mul_ps(row2_2, tmp1), minor1);
//     minor2 = _mm256_sub_ps(minor2, _mm256_mul_ps(row1_2, tmp1));
// 
//     tmp1 = _mm256_mul_ps(row0_2, row2_2);
//     tmp1 = _mm256_permute_ps(tmp1, 0xB1);
//     minor1 = _mm256_add_ps(_mm256_mul_ps(row3_2, tmp1), minor1);
//     minor3 = _mm256_sub_ps(minor3, _mm256_mul_ps(row1_2, tmp1));
//     tmp1 = _mm256_permute_ps(tmp1, 0x4E);
//     minor1 = _mm256_sub_ps(minor1, _mm256_mul_ps(row3_2, tmp1));
//     minor3 = _mm256_add_ps(_mm256_mul_ps(row1_2, tmp1), minor3);
// 
//     __m256 det = _mm256_mul_ps(row0_2, minor0);
//     det = _mm256_add_ps(det, _mm256_permute_ps(det, 0x4E));
//     det = _mm256_add_ps(det, _mm256_permute_ps(det, 0xB1));
// 
//     tmp1 = _mm256_rcp_ps(det);
//     det = _mm256_sub_ps(_mm256_add_ps(tmp1, tmp1), _mm256_mul_ps(det, _mm256_mul_ps(tmp1, tmp1)));
//     det = _mm256_permute_ps(det, 0x00);
// 
//     minor0 = _mm256_mul_ps(det, minor0);
//     minor1 = _mm256_mul_ps(det, minor1);
//     minor2 = _mm256_mul_ps(det, minor2);
//     minor3 = _mm256_mul_ps(det, minor3);
// 
//     _mm256_storeu_ps(&B[0], minor0);
//     _mm256_storeu_ps(&B[4], minor1);
//     _mm256_storeu_ps(&B[8], minor2);
//     _mm256_storeu_ps(&B[12], minor3);
// }

using namespace squint;

auto main() -> int {
    // mat4 A = mat4::random(0, 1);
    // auto B = inv(A);
    // std::cout << B << std::endl;
    // mat4 C{};
    // mat4 a_t = A.transpose();
    // invert_matrix_4x4_avx(a_t.data(), C.data());
    // std::cout << C << std::endl;

    // create N random vec3s
    constexpr size_t N = 1000000;
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