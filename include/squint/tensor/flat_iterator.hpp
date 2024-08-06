#ifndef SQUINT_FLAT_ITERATOR_HPP
#define SQUINT_FLAT_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include <vector>
#include <numeric>

namespace squint {

// Flat iterator for tensors
template <typename TensorType>
class flat_iterator {
    using value_type = std::remove_const_t<typename TensorType::value_type>;
    TensorType* tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type*, value_type*>;
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type&, value_type&>;

    flat_iterator(TensorType* tensor, const std::vector<std::size_t>& indices)
        : tensor_(tensor), current_indices_(indices), shape_(tensor->shape()), strides_(tensor->strides()) {}

    reference operator*() const {
        return tensor_->data()[std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL)];
    }

    pointer operator->() const { return &(operator*()); }

    flat_iterator& operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < shape_[i]) return *this;
            current_indices_[i] = 0;
        }
        current_indices_ = std::vector<std::size_t>(shape_.begin(), shape_.end()); // Set to end
        return *this;
    }

    flat_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    flat_iterator& operator--() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (current_indices_[i]-- > 0) return *this;
            current_indices_[i] = shape_[i] - 1;
        }
        std::fill(current_indices_.begin(), current_indices_.end(), 0); // Set to begin
        return *this;
    }

    flat_iterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    flat_iterator& operator+=(difference_type n) {
        // Implement efficient addition using division and modulus
        std::size_t total_offset = std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL) + n;
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] = total_offset / strides_[i];
            total_offset %= strides_[i];
        }
        return *this;
    }

    flat_iterator& operator-=(difference_type n) { return *this += -n; }

    flat_iterator operator+(difference_type n) const {
        auto result = *this;
        result += n;
        return result;
    }

    flat_iterator operator-(difference_type n) const {
        auto result = *this;
        result -= n;
        return result;
    }

    difference_type operator-(const flat_iterator& other) const {
        return std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0LL) -
               std::inner_product(other.current_indices_.begin(), other.current_indices_.end(), other.strides_.begin(), 0LL);
    }

    reference operator[](difference_type n) const { return *(*this + n); }

    bool operator==(const flat_iterator& other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const flat_iterator& other) const { return !(*this == other); }
    bool operator<(const flat_iterator& other) const {
        return std::lexicographical_compare(current_indices_.begin(), current_indices_.end(),
                                            other.current_indices_.begin(), other.current_indices_.end());
    }
    bool operator>(const flat_iterator& other) const { return other < *this; }
    bool operator<=(const flat_iterator& other) const { return !(other < *this); }
    bool operator>=(const flat_iterator& other) const { return !(*this < other); }

    friend flat_iterator operator+(difference_type n, const flat_iterator& it) { return it + n; }
};

} // namespace squint

#endif // SQUINT_FLAT_ITERATOR_HPP