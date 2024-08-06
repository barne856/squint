#ifndef SQUINT_TENSOR_TENSOR_VIEW_HPP
#define SQUINT_TENSOR_TENSOR_VIEW_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/tensor/tensor_base.hpp"
#include "squint/tensor/iterable_tensor.hpp"
#include "squint/util/array_utils.hpp"
#include <array>
#include <vector>

namespace squint {

template <typename T, typename Shape, typename ParentTensor,
          layout Layout = layout::row_major,
          error_checking ErrorChecking = error_checking::disabled>
class tensor_view : public iterable_tensor<tensor_view<T, Shape, ParentTensor, Layout, ErrorChecking>,
                                           T, Shape, Layout, ErrorChecking> {
public:
    using value_type = T;
    using shape_type = Shape;
    using parent_type = ParentTensor;

private:
    ParentTensor& parent_;
    using index_type = std::conditional_t<fixed_shape<ParentTensor>, std::array<std::size_t, shape_type::size()>, std::vector<std::size_t>>;
    index_type offset_;
    index_type view_shape_;
    index_type view_strides_;

public:
    // Constructor
    template <typename OffsetType, typename ViewShapeType>
    tensor_view(ParentTensor& parent, const index_type& offset, const index_type& view_shape)
        : parent_(parent), offset_(offset), view_shape_(view_shape) {
        calculate_strides();
    }

    // Required methods for tensor_base
    static constexpr std::size_t rank() {
        if constexpr (is_index_sequence<Shape>::value) {
            return Shape::size();
        } else {
            return view_shape_.size();
        }
    }

    std::size_t size() const {
        if constexpr (is_index_sequence<Shape>::value) {
            return product(Shape{});
        } else {
            return std::accumulate(view_shape_.begin(), view_shape_.end(), 1ULL, std::multiplies<std::size_t>());
        }
    }

    auto shape() const -> decltype(view_shape_) {
        return view_shape_;
    }

    auto strides() const -> decltype(view_strides_) {
        return view_strides_;
    }

    T* data() { return parent_.data() + calculate_data_offset(); }
    const T* data() const { return parent_.data() + calculate_data_offset(); }

    // Element access
    template <typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) == rank() || !is_index_sequence<Shape>::value,
                      "Number of indices must match tensor rank for fixed shape tensors");
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::array<std::size_t, sizeof...(Indices)>{static_cast<std::size_t>(indices)...});
        }
        return *compute_element_pointer(indices...);
    }

    template <typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == rank() || !is_index_sequence<Shape>::value,
                      "Number of indices must match tensor rank for fixed shape tensors");
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::array<std::size_t, sizeof...(Indices)>{static_cast<std::size_t>(indices)...});
        }
        return *compute_element_pointer(indices...);
    }

    // Subview
    template <typename SubviewShape>
    auto subview(const SubviewShape& subview_shape, const SubviewShape& start_indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_subview_bounds(subview_shape, start_indices);
        }

        auto new_offset = offset_;
        for (size_t i = 0; i < rank(); ++i) {
            new_offset[i] += start_indices[i];
        }

        return tensor_view<T, SubviewShape, ParentTensor, Layout, ErrorChecking>(
            parent_, new_offset, subview_shape);
    }

    // Fixed shape subview
    template <std::size_t... Dims>
    auto subview(const std::array<std::size_t, sizeof...(Dims)>& start_indices) const {
        return subview(std::index_sequence<Dims...>{}, start_indices);
    }

private:
    void calculate_strides() {
        view_strides_.resize(rank());
        if constexpr (Layout == layout::row_major) {
            view_strides_.back() = 1;
            for (int i = rank() - 2; i >= 0; --i) {
                view_strides_[i] = view_strides_[i + 1] * view_shape_[i + 1];
            }
        } else {
            view_strides_.front() = 1;
            for (size_t i = 1; i < rank(); ++i) {
                view_strides_[i] = view_strides_[i - 1] * view_shape_[i - 1];
            }
        }
    }

    std::size_t calculate_data_offset() const {
        std::size_t offset = 0;
        for (size_t i = 0; i < rank(); ++i) {
            offset += offset_[i] * parent_.strides()[i];
        }
        return offset;
    }

    template <typename... Indices>
    T* compute_element_pointer(Indices... indices) const {
        std::size_t index = 0;
        size_t i = 0;
        ((index += static_cast<std::size_t>(indices) * view_strides_[i++]), ...);
        return data() + index;
    }

    template <typename Indices>
    void check_bounds(const Indices& indices) const {
        for (size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= view_shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    template <typename SubviewShape>
    void check_subview_bounds(const SubviewShape& subview_shape, const SubviewShape& start_indices) const {
        for (size_t i = 0; i < rank(); ++i) {
            if (start_indices[i] + subview_shape[i] > view_shape_[i]) {
                throw std::out_of_range("Subview out of bounds");
            }
        }
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_VIEW_HPP