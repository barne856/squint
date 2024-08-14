#ifndef SQUINT_TENSOR_TENSOR_ITERATION_HPP
#define SQUINT_TENSOR_TENSOR_ITERATION_HPP

#include "squint/tensor/flat_iterator.hpp"
#include "squint/tensor/subview_iterator.hpp"
#include "squint/tensor/tensor.hpp"
#include "squint/tensor/tensor_op_compatibility.hpp"
#include "squint/util/sequence_utils.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <vector>

namespace squint {

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::rows() {
    if constexpr (fixed_shape<Shape>) {
        using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
        return this->template subviews<RowShape>();
    } else {
        std::vector<std::size_t> row_shape = this->shape();
        row_shape[0] = 1;
        return this->subviews(row_shape);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::rows() const {
    if constexpr (fixed_shape<Shape>) {
        using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
        return this->template subviews<RowShape>();
    } else {
        std::vector<std::size_t> row_shape = this->shape();
        row_shape[0] = 1;
        return this->subviews(row_shape);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::cols() {
    if constexpr (fixed_shape<Shape>) {
        using ColShape = init_sequence_t<Shape>;
        return this->template subviews<ColShape>();
    } else {
        std::vector<size_t> col_shape = this->shape();
        col_shape.pop_back();
        return this->subviews(col_shape);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::cols() const {
    if constexpr (fixed_shape<Shape>) {
        using ColShape = init_sequence_t<Shape>;
        return this->template subviews<ColShape>();
    } else {
        std::vector<size_t> col_shape = this->shape();
        col_shape.pop_back();
        return this->subviews(col_shape);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::row(size_t index) {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (index >= std::get<0>(make_array(Shape{}))) {
            throw std::out_of_range("Row index out of range");
        }
    }

    if constexpr (fixed_shape<Shape>) {
        using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
        return tensor<T, RowShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data() + index * std::get<0>(make_array(Strides{})));
    } else {
        std::vector<size_t> row_shape = this->shape();
        row_shape[0] = 1;
        return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data() + index * this->strides()[0], row_shape, this->strides());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::row(size_t index) const {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (index >= std::get<0>(make_array(Shape{}))) {
            throw std::out_of_range("Row index out of range");
        }
    }

    if constexpr (fixed_shape<Shape>) {
        using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
        return tensor<const T, RowShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data() + index * std::get<0>(make_array(Strides{})));
    } else {
        std::vector<size_t> row_shape = this->shape();
        row_shape[0] = 1;
        return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data() + index * this->strides()[0], row_shape, this->strides());
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::col(size_t index) {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (index >= std::get<1>(make_array(Shape{}))) {
            throw std::out_of_range("Column index out of range");
        }
    }

    if constexpr (fixed_shape<Shape>) {
        using ColShape = init_sequence_t<Shape>;
        using ColStrides = init_sequence_t<Strides>;
        constexpr std::size_t N = ColStrides::size();
        return tensor<T, ColShape, ColStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data() + index * std::get<N>(make_array(Strides{})));
    } else {
        std::vector<size_t> col_shape = this->shape();
        col_shape.pop_back();
        std::vector<size_t> col_strides = this->strides();
        col_strides.pop_back();
        std::size_t N = col_strides.size();
        return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data() + index * this->strides()[N], col_shape, col_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::col(size_t index) const {
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (index >= std::get<1>(make_array(Shape{}))) {
            throw std::out_of_range("Column index out of range");
        }
    }

    if constexpr (fixed_shape<Shape>) {
        using ColShape = init_sequence_t<Shape>;
        using ColStrides = init_sequence_t<Strides>;
        constexpr std::size_t N = ColStrides::size();
        return tensor<const T, ColShape, ColStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data() + index * std::get<N>(make_array(Strides{})));
    } else {
        std::vector<size_t> col_shape = this->shape();
        col_shape.pop_back();
        std::vector<size_t> col_strides = this->strides();
        col_strides.pop_back();
        std::size_t N = col_strides.size();
        return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(this->data() + index * this->strides()[N], col_shape, col_strides);
    }
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::begin() -> flat_iterator<tensor> {
    typename tensor::index_type start_indices{};
    if constexpr (dynamic_shape<Shape>) {
        start_indices.resize(this->rank(), 0);
    }
    return flat_iterator<tensor>(this, start_indices);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::end() -> flat_iterator<tensor> {
    return flat_iterator<tensor>(this, this->shape());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::begin() const
    -> flat_iterator<const tensor> {
    typename tensor::index_type start_indices{};
    if constexpr (dynamic_shape<Shape>) {
        start_indices.resize(this->rank(), 0);
    }
    return flat_iterator<const tensor>(this, start_indices);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::end() const -> flat_iterator<const tensor> {
    return flat_iterator<const tensor>(this, this->shape());
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::cbegin() const
    -> flat_iterator<const tensor> {
    return begin();
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::cend() const -> flat_iterator<const tensor> {
    return end();
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <fixed_shape SubviewShape>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews()
    -> iterator_range<subview_iterator<tensor, SubviewShape>>
    requires fixed_shape<Shape>
{
    static_assert(SubviewShape::size() <= Shape::size(),
                  "Subview dimensions must be less than or equal to tensor rank");
    static_assert(subview_compatible<tensor, SubviewShape>(),
                  "Subview dimensions must evenly divide tensor dimensions");
    constexpr auto end_indices = make_array(Shape{});
    auto subview_shape = make_array(Shape{});
    for (std::size_t i = 0; i < Shape::size(); ++i) {
        if (i < SubviewShape::size()) {
            subview_shape[i] = make_array(SubviewShape{})[i];
        } else {
            subview_shape[i] = 1;
        }
    }
    auto begin = subview_iterator<tensor, SubviewShape>(this, std::array<std::size_t, Shape::size()>{}, subview_shape);
    auto end = subview_iterator<tensor, SubviewShape>(
        this,
        [end_indices, subview_shape] {
            std::array<std::size_t, Shape::size()> result;
            std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), result.begin(),
                           std::divides<>());
            return result;
        }(),
        subview_shape);
    return iterator_range(begin, end);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <std::size_t... Dims>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews()
    -> iterator_range<subview_iterator<tensor, std::index_sequence<Dims...>>>
    requires fixed_shape<Shape>
{
    return subviews<std::index_sequence<Dims...>>();
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <fixed_shape SubviewShape>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews() const
    -> iterator_range<subview_iterator<const tensor, SubviewShape>>
    requires fixed_shape<Shape>
{
    static_assert(SubviewShape::size() <= Shape::size(),
                  "Subview dimensions must be less than or equal to tensor rank");
    static_assert(subview_compatible<tensor, SubviewShape>(),
                  "Subview dimensions must evenly divide tensor dimensions");
    constexpr auto end_indices = make_array(Shape{});
    auto subview_shape = make_array(Shape{});
    for (std::size_t i = 0; i < Shape::size(); ++i) {
        if (i < SubviewShape::size()) {
            subview_shape[i] = make_array(SubviewShape{})[i];
        } else {
            subview_shape[i] = 1;
        }
    }
    auto begin =
        subview_iterator<const tensor, SubviewShape>(this, std::array<std::size_t, Shape::size()>{}, subview_shape);
    auto end = subview_iterator<const tensor, SubviewShape>(
        this,
        [end_indices, subview_shape] {
            std::array<std::size_t, Shape::size()> result;
            std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), result.begin(),
                           std::divides<>());
            return result;
        }(),
        subview_shape);
    return iterator_range(begin, end);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
template <std::size_t... Dims>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews() const
    -> iterator_range<subview_iterator<const tensor, std::index_sequence<Dims...>>>
    requires fixed_shape<Shape>
{
    return subviews<std::index_sequence<Dims...>>();
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews(
    const std::vector<std::size_t> &subview_shape) -> iterator_range<subview_iterator<tensor, std::vector<std::size_t>>>
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (subview_shape.size() > this->rank()) {
            throw std::invalid_argument("Subview dimensions must be less than or equal to tensor rank");
        }
        for (std::size_t i = 0; i < subview_shape.size(); ++i) {
            if (this->shape()[i] % subview_shape[i] != 0) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }
    }
    // add trailing 1s to subview shape
    auto subview_shape_copy = subview_shape;
    subview_shape_copy.resize(this->rank(), 1);

    auto begin = subview_iterator<tensor, std::vector<std::size_t>>(this, std::vector<std::size_t>(this->rank(), 0),
                                                                    subview_shape_copy);
    auto end = subview_iterator<tensor, std::vector<std::size_t>>(
        this,
        [this, &subview_shape_copy] {
            auto end_indices = this->shape();
            std::transform(end_indices.begin(), end_indices.end(), subview_shape_copy.begin(), end_indices.begin(),
                           std::divides<>());
            return end_indices;
        }(),
        subview_shape_copy);
    return iterator_range(begin, end);
}

template <typename T, typename Shape, typename Strides, error_checking ErrorChecking, ownership_type OwnershipType,
          memory_space MemorySpace>
auto tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>::subviews(
    const std::vector<std::size_t> &subview_shape) const
    -> iterator_range<subview_iterator<const tensor, std::vector<std::size_t>>>
    requires dynamic_shape<Shape>
{
    if constexpr (ErrorChecking == error_checking::enabled) {
        if (subview_shape.size() > this->rank()) {
            throw std::invalid_argument("Subview dimensions must be less than or equal to tensor rank");
        }
        for (std::size_t i = 0; i < subview_shape.size(); ++i) {
            if (this->shape()[i] % subview_shape[i] != 0) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }
    }
    // add trailing 1s to subview shape
    auto subview_shape_copy = subview_shape;
    subview_shape_copy.resize(this->rank(), 1);

    auto begin = subview_iterator<const tensor, std::vector<std::size_t>>(
        this, std::vector<std::size_t>(this->rank(), 0), subview_shape_copy);
    auto end = subview_iterator<const tensor, std::vector<std::size_t>>(
        this,
        [this, &subview_shape_copy] {
            auto end_indices = this->shape();
            std::transform(end_indices.begin(), end_indices.end(), subview_shape_copy.begin(), end_indices.begin(),
                           std::divides<>());
            return end_indices;
        }(),
        subview_shape_copy);
    return iterator_range(begin, end);
}

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_ITERATION_HPP