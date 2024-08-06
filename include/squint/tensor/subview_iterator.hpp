#ifndef SQUINT_SUBVIEW_ITERATOR_HPP
#define SQUINT_SUBVIEW_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include <vector>

namespace squint {

// Base class for subview iterators
template <typename TensorType> class subview_iterator_base {
  protected:
    using value_type = std::remove_const_t<typename TensorType::value_type>;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> subview_shape_;
    std::vector<std::size_t> tensor_shape_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    subview_iterator_base(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                          const std::vector<std::size_t> &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape),
          tensor_shape_(tensor->shape()) {}

    subview_iterator_base &operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i])
                return *this;
            current_indices_[i] = 0;
        }
        // Set to end
        current_indices_ = std::vector<std::size_t>(tensor_shape_.begin(), tensor_shape_.end());
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            current_indices_[i] /= subview_shape_[i];
        }
        return *this;
    }

    bool operator==(const subview_iterator_base &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const subview_iterator_base &other) const { return !(*this == other); }

  protected:
    std::vector<std::size_t> get_offset() const {
        std::vector<std::size_t> start(current_indices_.size());
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            start[i] = current_indices_[i] * subview_shape_[i];
        }
        return start;
    }
};

// Subview iterator for fixed shape tensors
template <typename TensorType, std::size_t... SubviewDims>
class fixed_subview_iterator : public subview_iterator_base<TensorType> {
    using Base = subview_iterator_base<TensorType>;

  public:
    fixed_subview_iterator(TensorType *tensor, const std::array<std::size_t, sizeof...(SubviewDims)> &start_indices)
        : Base(tensor, std::vector<std::size_t>(start_indices.begin(), start_indices.end()),
               std::vector<std::size_t>{SubviewDims...}) {}

    auto operator*() const {
        return [this]<std::size_t... Is>(std::index_sequence<Is...>) {
            return this->tensor_->template subview<SubviewDims...>(
                (this->current_indices_[Is] * this->subview_shape_[Is])...);
        }(std::make_index_sequence<sizeof...(SubviewDims)>{});
    }

    fixed_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    fixed_subview_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
};

// Subview iterator for dynamic shape tensors
template <typename TensorType> class dynamic_subview_iterator : public subview_iterator_base<TensorType> {
    using Base = subview_iterator_base<TensorType>;

  public:
    dynamic_subview_iterator(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                             const std::vector<std::size_t> &subview_shape)
        : Base(tensor, start_indices, subview_shape) {}

    auto operator*() const { return this->tensor_->subview(this->subview_shape_, this->get_offset()); }

    dynamic_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    dynamic_subview_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
};

} // namespace squint

#endif // SQUINT_SUBVIEW_ITERATOR_HPP