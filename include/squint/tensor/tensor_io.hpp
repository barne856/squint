#ifndef SQUINT_TENSOR_TENSOR_IO_HPP
#define SQUINT_TENSOR_TENSOR_IO_HPP

#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"

#include <iomanip>
#include <iostream>

namespace squint {

// Helper function to print a 2D slice of a tensor
template <typename TensorType> void print_slice(std::ostream &os, const TensorType &t) {
    const auto &shape = t.shape();
    const auto rows = shape[0];
    const auto cols = shape.size() > 1 ? shape[1] : 1;

    for (size_t i = 0; i < rows; ++i) {
        os << "[";
        for (size_t j = 0; j < cols; ++j) {
            constexpr std::size_t width = 8;
            os << std::setw(width) << std::setprecision(4) << t(i, j);
            if (j < cols - 1) {
                os << ", ";
            }
        }
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
            print_slice(os, t.template reshape<std::get<0>(make_array(Shape{})), 1>());
        } else if constexpr (Shape::size() == 2) {
            print_slice(os, t);
        } else {
            std::size_t slice_index = 0;
            for (const auto &slice :
                 t.template subviews<std::get<0>(make_array(Shape{})), std::get<1>(make_array(Shape{}))>()) {
                os << "Slice " << slice_index++ << ":\n";
                print_slice(os, slice);
                os << "\n";
            }
        }
    } else {
        if (rank == 1) {
            // Print as a column vector
            print_slice(os, t.reshape({t.shape()[0], 1}));
        } else if (rank == 2) {
            print_slice(os, t);
        } else {
            std::size_t slice_index = 0;
            for (const auto &slice : t.subviews({shape[0], shape[1]})) {
                os << "Slice " << slice_index++ << ":\n";
                print_slice(os, slice);
                os << "\n";
            }
        }
    }
    return os;
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_IO_HPP