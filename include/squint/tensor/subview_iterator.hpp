#ifndef SQUINT_SUBVIEW_ITERATOR_HPP
#define SQUINT_SUBVIEW_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/util/array_utils.hpp"

namespace squint {

template <typename TensorType, typename SubviewShape> class subview_iterator {
    TensorType *tensor_;
    using value_type = typename TensorType::value_type;
    using index_type = typename TensorType::index_type;
    index_type current_indices_;
    index_type subview_shape_;
    index_type tensor_shape_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    subview_iterator(TensorType *tensor, const index_type &start_indices, const index_type &subview_shape)
        requires dynamic_shape<TensorType>
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape),
          tensor_shape_(tensor->shape()) {}

    subview_iterator(TensorType *tensor, const index_type &start_indices)
        requires fixed_shape<TensorType>
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(make_array(SubviewShape{})),
          tensor_shape_(make_array(typename TensorType::shape_type{})) {}

    auto operator++() -> subview_iterator & {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        // Set to end
        current_indices_ = tensor_shape_;
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            current_indices_[i] /= subview_shape_[i];
        }
        return *this;
    }

    auto operator++(int) -> subview_iterator {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    auto operator*() const {
        if constexpr (fixed_shape<TensorType>) {
            return this->tensor_->template subview<SubviewShape>(this->get_offset());
        } else {
            return this->tensor_->subview(this->subview_shape_, this->get_offset());
        }
    }

    auto operator==(const subview_iterator &other) const -> bool {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    auto operator!=(const subview_iterator &other) const -> bool { return !(*this == other); }

  private:
    auto get_offset() const -> index_type {
        index_type start{};
        if constexpr (dynamic_shape<TensorType>) {
            start = index_type(current_indices_.size());
        }
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            start[i] = current_indices_[i] * subview_shape_[i];
        }
        return start;
    }
};

} // namespace squint

#endif // SQUINT_SUBVIEW_ITERATOR_HPP