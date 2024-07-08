#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/tensor_base.hpp"
#include <array>
#include <iterator>
#include <vector>

namespace squint {

// Flat iterator for tensors
template <typename TensorType> class flat_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> strides_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    flat_iterator(TensorType *tensor, const std::vector<std::size_t> &indices)
        : tensor_(tensor), current_indices_(indices) {
        if constexpr (fixed_shape_tensor<TensorType>) {
            constexpr auto tensor_strides = TensorType::constexpr_strides();
            strides_.assign(tensor_strides.begin(), tensor_strides.end());
        } else {
            strides_ = tensor->strides();
        }
    }

    reference operator*() const {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            offset += current_indices_[i] * strides_[i];
        }
        return tensor_->data()[offset];
    }

    pointer operator->() const { return &(operator*()); }

    flat_iterator &operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            ++current_indices_[i];
            if (current_indices_[i] < tensor_->shape()[i]) {
                break;
            }
            if (i > 0) {
                current_indices_[i] = 0;
            }
        }
        return *this;
    }

    flat_iterator operator++(int) {
        flat_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    flat_iterator &operator--() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (current_indices_[i] > 0) {
                --current_indices_[i];
                break;
            }
            if (i > 0) {
                current_indices_[i] = tensor_->shape()[i] - 1;
            }
        }
        return *this;
    }

    flat_iterator operator--(int) {
        flat_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    flat_iterator &operator+=(difference_type n) {
        // Implement addition logic
        // This is a simplification and might not work correctly for all tensor shapes
        std::size_t total_size = tensor_->size();
        std::size_t current_index = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            current_index += current_indices_[i] * tensor_->strides()[i];
        }
        current_index = (current_index + n) % total_size;

        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] = current_index / tensor_->strides()[i];
            current_index %= tensor_->strides()[i];
        }
        return *this;
    }

    flat_iterator &operator-=(difference_type n) { return operator+=(-n); }

    flat_iterator operator+(difference_type n) const {
        flat_iterator result = *this;
        result += n;
        return result;
    }

    flat_iterator operator-(difference_type n) const {
        flat_iterator result = *this;
        result -= n;
        return result;
    }

    difference_type operator-(const flat_iterator &other) const {
        // Implement subtraction logic
        // This is a simplification and might not work correctly for all tensor shapes
        std::size_t this_index = 0;
        std::size_t other_index = 0;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            this_index += current_indices_[i] * tensor_->strides()[i];
            other_index += other.current_indices_[i] * tensor_->strides()[i];
        }
        return static_cast<difference_type>(this_index) - static_cast<difference_type>(other_index);
    }

    reference operator[](difference_type n) const { return *(*this + n); }

    bool operator==(const flat_iterator &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const flat_iterator &other) const { return !(*this == other); }
    bool operator<(const flat_iterator &other) const { return (other - *this) > 0; }
    bool operator>(const flat_iterator &other) const { return other < *this; }
    bool operator<=(const flat_iterator &other) const { return !(other < *this); }
    bool operator>=(const flat_iterator &other) const { return !(*this < other); }
};

// Fixed subview iterator for tensors
template <typename TensorType, std::size_t... SubviewDims> class fixed_subview_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::array<std::size_t, sizeof...(SubviewDims)> current_indices_;
    std::array<std::size_t, sizeof...(SubviewDims)> subview_shape_;
    std::array<std::size_t, sizeof...(SubviewDims)> strides_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    fixed_subview_iterator(TensorType *tensor, const std::array<std::size_t, sizeof...(SubviewDims)> &start_indices)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_{SubviewDims...} {
        if constexpr (fixed_shape_tensor<TensorType>) {
            constexpr auto tensor_strides = TensorType::constexpr_strides();
            std::copy(tensor_strides.begin(), tensor_strides.end(), strides_.begin());
        } else {
            auto tensor_strides = tensor->strides();
            std::copy(tensor_strides.begin(), tensor_strides.end(), strides_.begin());
        }
    }

    auto operator*() const { return make_subview(std::make_index_sequence<sizeof...(SubviewDims)>{}); }

    fixed_subview_iterator &operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] += subview_shape_[i];
            if (current_indices_[i] < tensor_->shape()[i]) {
                break;
            }
            if (i > 0) {
                current_indices_[i] = 0;
            } else {
                // We've reached the end of the tensor
                // Set all indices to their end positions
                for (size_t j = 0; j < current_indices_.size(); ++j) {
                    current_indices_[j] = tensor_->shape()[j] - (tensor_->shape()[j] % subview_shape_[j]);
                }
                break;
            }
        }
        return *this;
    }

    fixed_subview_iterator operator++(int) {
        fixed_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const fixed_subview_iterator &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const fixed_subview_iterator &other) const { return !(*this == other); }

  private:
    template <std::size_t... Is> auto make_subview(std::index_sequence<Is...>) const {
        return tensor_->view().template subview<SubviewDims...>(slice{current_indices_[Is], subview_shape_[Is]}...);
    }
};

// Dynamic subview iterator for tensors
template <typename TensorType> class dynamic_subview_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> subview_shape_;
    std::vector<std::size_t> strides_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    dynamic_subview_iterator(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                             const std::vector<std::size_t> &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape), strides_(tensor->strides()) {
    }

    auto operator*() const {
        std::vector<slice> slices;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            slices.push_back({current_indices_[i], subview_shape_[i]});
        }
        return tensor_->subview(slices);
    }

    dynamic_subview_iterator &operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] += subview_shape_[i];
            if (current_indices_[i] < tensor_->shape()[i]) {
                break;
            }
            if (i > 0) {
                current_indices_[i] = 0;
            } else {
                // We've reached the end of the tensor
                // Set all indices to their end positions
                for (size_t j = 0; j < current_indices_.size(); ++j) {
                    current_indices_[j] = tensor_->shape()[j] - (tensor_->shape()[j] % subview_shape_[j]);
                }
                break;
            }
        }
        return *this;
    }

    dynamic_subview_iterator operator++(int) {
        dynamic_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const dynamic_subview_iterator &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const dynamic_subview_iterator &other) const { return !(*this == other); }
};

// Mixin class for iterable tensors
template <typename Derived, typename T> class iterable_tensor : public tensor_base<Derived, T> {
  public:
    using iterator = flat_iterator<Derived>;
    using const_iterator = flat_iterator<const Derived>;

    iterator begin() { return iterator(static_cast<Derived *>(this), std::vector<std::size_t>(this->rank(), 0)); }

    iterator end() {
        auto shape = this->shape();
        shape.back() = 0; // Set the last dimension to 0 to represent the end
        return iterator(static_cast<Derived *>(this), shape);
    }

    const_iterator begin() const {
        return const_iterator(static_cast<const Derived *>(this), std::vector<std::size_t>(this->rank(), 0));
    }

    const_iterator end() const {
        auto shape = this->shape();
        shape.back() = 0; // Set the last dimension to 0 to represent the end
        return const_iterator(static_cast<const Derived *>(this), shape);
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // Subview iteration for fixed shape tensors
    template <std::size_t... SubviewDims> auto subviews() {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");

        struct subview_range {
            Derived *tensor;

            auto begin() {
                return fixed_subview_iterator<Derived, SubviewDims...>(
                    tensor, std::array<std::size_t, sizeof...(SubviewDims)>());
            }

            auto end() {
                std::array<std::size_t, sizeof...(SubviewDims)> end_indices;
                auto shape = tensor->shape();
                std::size_t i = 0;
                ((end_indices[i] = shape[i] - (shape[i] % SubviewDims), ++i), ...);
                return fixed_subview_iterator<Derived, SubviewDims...>(tensor, end_indices);
            }
        };

        return subview_range{static_cast<Derived *>(this)};
    }

    // Subview iteration for dynamic shape tensors
    auto subviews(const std::vector<std::size_t> &subview_shape) {
        if (subview_shape.size() != this->rank()) {
            throw std::invalid_argument("Subview dimensions must match tensor rank");
        }

        struct subview_range {
            Derived *tensor;
            std::vector<std::size_t> subview_shape;

            auto begin() {
                return dynamic_subview_iterator<Derived>(tensor, std::vector<std::size_t>(tensor->rank(), 0),
                                                         subview_shape);
            }

            auto end() {
                std::vector<std::size_t> end_indices = tensor->shape();
                for (size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] -= (end_indices[i] % subview_shape[i]);
                }
                return dynamic_subview_iterator<Derived>(tensor, end_indices, subview_shape);
            }
        };

        return subview_range{static_cast<Derived *>(this), subview_shape};
    }
};

} // namespace squint

#endif // SQUINT_ITERABLE_TENSOR_HPP