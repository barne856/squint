#include "squint/fixed_tensor.hpp"
#include "squint/tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;

    fixed_tensor<double, layout::column_major, 4, 4> A;
    A[0, 0] = 1;
    A[0, 1] = 2;
    A[1, 0] = 3;
    A[1, 1] = 4;
    A[2, 0] = 5;
    A[2, 1] = 6;
    A[3, 0] = 7;
    A[3, 1] = 8;
    A[0, 2] = 9;
    A[0, 3] = 10;
    A[1, 2] = 11;
    A[1, 3] = 12;
    A[2, 2] = 13;
    A[2, 3] = 14;
    A[3, 2] = 15;
    A[3, 3] = 16;
    std::cout << "A:\n" << A << '\n';
    auto B = A.view();
    std::cout << "B:\n" << B << '\n';
    auto C = B.subview<2, 2>(slice{2, 2}, slice{2, 2});
    std::cout << "C:\n" << C << '\n';
    for (const auto &s : C.strides()) {
        std::cout << s << ' ';
    }
    std::cout << '\n';

    for (const auto &value : A) {
        std::cout << value << ' ';
    }

    for (const auto &value : B) {
        std::cout << value << ' ';
    }

    dynamic_tensor<double> Ad({4, 4}, layout::column_major);
    Ad[0, 0] = 1;
    Ad[0, 1] = 2;
    Ad[1, 0] = 3;
    Ad[1, 1] = 4;
    Ad[2, 0] = 5;
    Ad[2, 1] = 6;
    Ad[3, 0] = 7;
    Ad[3, 1] = 8;
    Ad[0, 2] = 9;
    Ad[0, 3] = 10;
    Ad[1, 2] = 11;
    Ad[1, 3] = 12;
    Ad[2, 2] = 13;
    Ad[2, 3] = 14;
    Ad[3, 2] = 15;
    Ad[3, 3] = 16;
    std::cout << "Ad:\n" << Ad << '\n';
    auto Bd = Ad.view();
    std::cout << "Bd:\n" << Bd << '\n';
    auto Cd = Bd.subview(slice{2, 2}, slice{2, 2});
    std::cout << "Cd:\n" << Cd << '\n';
    for (const auto &s : Cd.strides()) {
        std::cout << s << ' ';
    }
    std::cout << '\n';

    for (const auto &value : Ad) {
        std::cout << value << ' ';
    }
    for (const auto &value : Bd) {
        std::cout << value << ' ';
    }

    return 0;
}