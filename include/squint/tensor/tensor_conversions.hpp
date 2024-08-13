#ifndef SQUINT_TENSOR_TENSOR_CONVERSIONS_HPP
#define SQUINT_TENSOR_TENSOR_CONVERSIONS_HPP

#include "squint/tensor/tensor.hpp"
#include "squint/util/sequence_utils.hpp"
#include <array>
#include <vector>

namespace squint {
// implicit conversion to std::array
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator std::array<T, product(Shape{})>() const
    requires(fixed_shape<Shape>)
{
    // must be 1D
    static_assert(max(Shape{}) == product(Shape{}), "Cannot convert non-1D tensor to std::array");
    std::array<T, product(Shape{})> result;
    std::copy(begin(), end(), result.begin());
    return result;
}

// implicit conversion to std::vector
template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::operator std::vector<T>() const
    requires(dynamic_shape<Shape>)
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        std::size_t max = 0;
        std::size_t product = 1;
        for (std::size_t i = 0; i < Shape::size(); ++i) {
            if (shape_[i] > max) {
                max = shape_[i];
            }
            product *= shape_[i];
        }
        if (max != product) {
            throw std::runtime_error("Cannot convert non-1D tensor to std::vector");
        }
    }
    std::vector<T> result;
    result.resize(this->size());
    std::copy(begin(), end(), result.begin());
    return result;
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_CONVERSIONS_HPP