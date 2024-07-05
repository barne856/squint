#include "squint/tensor.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace squint;

    // Test fixed_tensor
    std::cout << "Testing fixed_tensor:\n";
    fixed_tensor<double, layout::column_major, 2, 3> fixed_matrix;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            fixed_matrix[i, j] = i * 3 + j + 1;
        }
    }
    std::cout << "Fixed Matrix:\n";
    std::cout << fixed_matrix << std::endl;

    // Test dynamic_tensor
    std::cout << "Testing dynamic_tensor:\n";
    dynamic_tensor<int> dynamic_tensor({3, 3, 3});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                dynamic_tensor[i, j, k] = i * 9 + j * 3 + k + 1;
            }
        }
    }
    std::cout << "Dynamic Tensor:\n";
    std::cout << dynamic_tensor << std::endl;

    // // Test tensor_view
    // std::cout << "Testing tensor_view:\n";
    // auto view = make_tensor_view(fixed_matrix);
    // print_tensor(view, "Tensor View of Fixed Matrix");
    //
    // // Test make_matrix_from_rows
    // std::cout << "Testing make_matrix_from_rows:\n";
    // auto row_matrix = make_matrix_from_rows<double, 2, 3>({{{1, 2, 3}, {4, 5, 6}}});
    // print_tensor(row_matrix, "Matrix from Rows");
    //
    // // Test make_matrix_from_columns
    // std::cout << "Testing make_matrix_from_columns:\n";
    // auto col_matrix = make_matrix_from_columns<double, 2, 3>({{{1, 4}, {2, 5}, {3, 6}}});
    // print_tensor(col_matrix, "Matrix from Columns");
    //
    // // Test sub_matrix
    // std::cout << "Testing sub_matrix:\n";
    // auto sub = sub_matrix<double, layout::row_major, 2, 3, 2, 2>(fixed_matrix, 0, 0);
    // print_tensor(sub, "Sub-matrix");
    //
    // // Test transposition (assuming it's implemented in linear_algebra_mixin)
    // std::cout << "Testing transpose:\n";
    // auto transposed = fixed_matrix.transpose();
    // print_tensor(transposed, "Transposed Fixed Matrix");

    return 0;
}