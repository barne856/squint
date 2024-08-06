#ifndef SQUINT_SUBVIEW_ITERATOR_HPP
#define SQUINT_SUBVIEW_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/util/array_utils.hpp"
#include <array>
#include <vector>

namespace squint {

template <typename TensorType>
class subview_iterator {
  private:
    using value_type = std::remove_const_t<typename TensorType::value_type>;
    using shape_type = typename TensorType::shape_type;
    TensorType* tensor_;
    using index_type = std::conditional_t<fixed_shape<TensorType>, std::array<std::size_t, shape_type::size()>, std::vector<std::size_t>>;
    index_type current_indices_;
    index_type subview_shape_;
    index_type tensor_shape_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type*, value_type*>;
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type&, value_type&>;

    subview_iterator(TensorType* tensor, const index_type& start_indices, const index_type& subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape) {
        if constexpr (fixed_shape<TensorType>) {
            tensor_shape_ = make_array(shape_type{});
        } else {
            tensor_shape_ = tensor->shape();
        }
    }

    subview_iterator& operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i])
                return *this;
            current_indices_[i] = 0;
        }
        // Set to end
        if constexpr (fixed_shape<TensorType>) {
            current_indices_ = tensor_shape_;
            for (size_t i = 0; i < current_indices_.size(); ++i) {
                current_indices_[i] /= subview_shape_[i];
            }
        } else {
            current_indices_ = std::vector<std::size_t>(tensor_shape_.begin(), tensor_shape_.end());
            for (size_t i = 0; i < current_indices_.size(); ++i) {
                current_indices_[i] /= subview_shape_[i];
            }
        }
        return *this;
    }

    subview_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    auto operator*() const {
        if constexpr (fixed_shape<TensorType>) {
            return [this]<std::size_t... Is>(std::index_sequence<Is...>) {
                return this->tensor_->template subview<std::index_sequence<subview_shape_[Is]...>>(
                    std::array<std::size_t, sizeof...(Is)>{(this->current_indices_[Is] * this->subview_shape_[Is])...});
            }(std::make_index_sequence<shape_type::size()>{});
        } else {
            return this->tensor_->subview(this->subview_shape_, this->get_offset());
        }
    }

    bool operator==(const subview_iterator& other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const subview_iterator& other) const { return !(*this == other); }

  private:
    std::vector<std::size_t> get_offset() const {
        std::vector<std::size_t> start(current_indices_.size());
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            start[i] = current_indices_[i] * subview_shape_[i];
        }
        return start;
    }
};

} // namespace squint

#endif // SQUINT_SUBVIEW_ITERATOR_HPP