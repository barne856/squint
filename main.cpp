#include "squint/dimension.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor.hpp"
#include <iostream>

using namespace squint;
using namespace squint::units;

int main() {
    // Test overdetermined system: 4 equations, 3 unknowns
    {
        auto A = fixed_tensor<double, layout::row_major, error_checking::disabled, 4, 3>::random();
        auto b = fixed_tensor<double, layout::row_major, error_checking::disabled, 4>::random();

        std::cout << "Overdetermined system:" << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "b = " << b << std::endl;

        auto x = solve_lls(A, b);
        std::cout << "Solution x = " << x << std::endl << std::endl;
    }

    // Test underdetermined system: 3 equations, 4 unknowns
    {
        auto A = fixed_tensor<double, layout::row_major, error_checking::disabled, 3, 4>::random();
        auto b = fixed_tensor<double, layout::row_major, error_checking::disabled, 3>::random();

        std::cout << "Underdetermined system:" << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "b = " << b << std::endl;

        auto x = solve_lls(A, b);
        std::cout << "Solution x = " << x << std::endl << std::endl;
    }

    // Test exactly determined system: 3 equations, 3 unknowns
    {
        auto A = fixed_tensor<double, layout::row_major, error_checking::disabled, 3, 3>::random();
        auto b = fixed_tensor<double, layout::row_major, error_checking::disabled, 3>::random();

        std::cout << "Exactly determined system:" << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "b = " << b << std::endl;

        auto x = solve_lls(A, b);
        std::cout << "Solution x = " << x << std::endl;
    }

    return 0;
}