#ifndef SQUINT_TENSOR_TENSOR_IO_HPP
#define SQUINT_TENSOR_TENSOR_IO_HPP

#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"

#include <iomanip>
#include <iostream>

namespace squint {

constexpr int SCALAR_WIDTH = 6;
constexpr int SCALAR_PRECISION = 4;

// Helper function to print a 2D slice of a tensor
template <typename TensorType> void print_2d_slice(std::ostream &os, const TensorType &t) {
    const auto &shape = t.shape();
    const auto rows = shape[0];
    const auto cols = shape.size() > 1 ? shape[1] : 1;

    for (size_t i = 0; i < rows; ++i) {
        os << "[";
        for (size_t j = 0; j < cols; ++j) {
            os << std::setw(SCALAR_WIDTH) << std::setprecision(SCALAR_PRECISION) << t(i, j);
            if (j < cols - 1) {
                os << ", ";
            }
        }
        os << "]\n";
    }
}

// Helper function to print a 1D slice of a tensor as column vector
template <typename TensorType> void print_1d_slice(std::ostream &os, const TensorType &t) {
    const auto &shape = t.shape();
    const auto size = shape[0];

    for (size_t i = 0; i < size; ++i) {
        os << "[";
        os << std::setw(SCALAR_WIDTH) << std::setprecision(SCALAR_PRECISION) << t(i);
        os << "]\n";
    }
}

// Overloaded stream output operator for tensor
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto operator<<(std::ostream &os,
                const tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace> &t) -> std::ostream & {
    const auto shape = t.shape();
    const auto rank = t.rank();

    // Print shape information
    os << "Tensor shape: [";
    for (size_t i = 0; i < rank; ++i) {
        os << shape[i];
        if (i < rank - 1) {
            os << ", ";
        }
    }
    os << "]\n";

    // Print as 2D slices
    if constexpr (fixed_shape<Shape>) {
        if constexpr (Shape::size() == 1) {
            // Print as a column vector
            print_1d_slice(os, t);
        } else if constexpr (Shape::size() == 2) {
            print_2d_slice(os, t);
        } else {
            std::size_t slice_index = 0;
            for (const auto &slice :
                 t.template subviews<std::get<0>(make_array(Shape{})), std::get<1>(make_array(Shape{}))>()) {
                os << "Slice " << slice_index++ << ":\n";
                print_2d_slice(os, slice);
                os << "\n";
            }
        }
    } else {
        if (rank == 1) {
            // Print as a column vector
            print_1d_slice(os, t);
        } else if (rank == 2) {
            print_2d_slice(os, t);
        } else {
            std::size_t slice_index = 0;
            for (const auto &slice : t.subviews({shape[0], shape[1]})) {
                os << "Slice " << slice_index++ << ":\n";
                print_2d_slice(os, slice);
                os << "\n";
            }
        }
    }
    return os;
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_IO_HPP