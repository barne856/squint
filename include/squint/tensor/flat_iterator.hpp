#ifndef SQUINT_FLAT_ITERATOR_HPP
#define SQUINT_FLAT_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/util/array_utils.hpp"
#include <numeric>
#include <type_traits>

namespace squint {

// Flat iterator for tensors
template <typename TensorType> class flat_iterator {
    TensorType *tensor_;
    using index_type = typename TensorType::index_type;
    using value_type = typename TensorType::value_type;
    index_type current_indices_;
    index_type shape_;
    index_type strides_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    flat_iterator(TensorType *tensor, const index_type &indices) : tensor_(tensor), current_indices_(indices) {
        if constexpr (fixed_shape<TensorType>) {
            shape_ = make_array(typename TensorType::shape_type{});
            strides_ = make_array(typename TensorType::stride_type{});
        } else {
            shape_ = tensor->shape();
            strides_ = tensor->strides();
        }
    }

    auto operator*() const -> reference {
        return tensor_
            ->data()[std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL)];
    }

    auto operator->() const -> pointer { return &(operator*()); }

    auto operator++() -> flat_iterator & {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        current_indices_ = shape_; // Set to end
        return *this;
    }

    auto operator++(int) -> flat_iterator {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    auto operator--() -> flat_iterator & {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (current_indices_[i]-- > 0) {
                return *this;
            }
            current_indices_[i] = shape_[i] - 1;
        }
        std::fill(current_indices_.begin(), current_indices_.end(), 0); // Set to begin
        return *this;
    }

    auto operator--(int) -> flat_iterator {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    auto operator+=(difference_type n) -> flat_iterator & {
        std::size_t total_offset =
            std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL) + n;
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] = total_offset / strides_[i];
            total_offset %= strides_[i];
        }
        return *this;
    }

    auto operator-=(difference_type n) -> flat_iterator & { return *this += -n; }

    auto operator+(difference_type n) const -> flat_iterator {
        auto result = *this;
        result += n;
        return result;
    }

    auto operator-(difference_type n) const -> flat_iterator {
        auto result = *this;
        result -= n;
        return result;
    }

    auto operator-(const flat_iterator &other) const -> difference_type {
        return std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0LL) -
               std::inner_product(other.current_indices_.begin(), other.current_indices_.end(), other.strides_.begin(),
                                  0LL);
    }

    auto operator[](difference_type n) const -> reference { return *(*this + n); }

    auto operator==(const flat_iterator &other) const -> bool {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    auto operator!=(const flat_iterator &other) const -> bool { return !(*this == other); }
    auto operator<(const flat_iterator &other) const -> bool {
        return std::lexicographical_compare(current_indices_.begin(), current_indices_.end(),
                                            other.current_indices_.begin(), other.current_indices_.end());
    }
    auto operator>(const flat_iterator &other) const -> bool { return other < *this; }
    auto operator<=(const flat_iterator &other) const -> bool { return !(other < *this); }
    auto operator>=(const flat_iterator &other) const -> bool { return !(*this < other); }

    friend auto operator+(difference_type n, const flat_iterator &it) -> flat_iterator { return it + n; }
};

} // namespace squint

#endif // SQUINT_FLAT_ITERATOR_HPP