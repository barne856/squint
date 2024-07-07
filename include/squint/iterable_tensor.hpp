#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/tensor_base.hpp"
#include <array>
#include <iterator>

namespace squint {

// Flat iterator for tensors
template <typename TensorType> class flat_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::size_t index_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    flat_iterator(TensorType *tensor, std::size_t index) : tensor_(tensor), index_(index) {}

    reference operator*() const { return tensor_->data()[index_]; }
    pointer operator->() const { return &tensor_->data()[index_]; }

    flat_iterator &operator++() {
        ++index_;
        return *this;
    }
    flat_iterator operator++(int) {
        flat_iterator tmp = *this;
        ++index_;
        return tmp;
    }

    flat_iterator &operator--() {
        --index_;
        return *this;
    }
    flat_iterator operator--(int) {
        flat_iterator tmp = *this;
        --index_;
        return tmp;
    }

    flat_iterator &operator+=(difference_type n) {
        index_ += n;
        return *this;
    }
    flat_iterator &operator-=(difference_type n) {
        index_ -= n;
        return *this;
    }

    flat_iterator operator+(difference_type n) const { return flat_iterator(tensor_, index_ + n); }
    flat_iterator operator-(difference_type n) const { return flat_iterator(tensor_, index_ - n); }

    difference_type operator-(const flat_iterator &other) const { return index_ - other.index_; }

    reference operator[](difference_type n) const { return tensor_->data()[index_ + n]; }

    bool operator==(const flat_iterator &other) const { return index_ == other.index_; }
    bool operator!=(const flat_iterator &other) const { return index_ != other.index_; }
    bool operator<(const flat_iterator &other) const { return index_ < other.index_; }
    bool operator>(const flat_iterator &other) const { return index_ > other.index_; }
    bool operator<=(const flat_iterator &other) const { return index_ <= other.index_; }
    bool operator>=(const flat_iterator &other) const { return index_ >= other.index_; }
};

// Subview iterator for tensors
template <typename TensorType, typename SubviewShape> class subview_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    SubviewShape subview_shape_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    subview_iterator(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                     const SubviewShape &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape) {}

    auto operator*() const {
        if constexpr (fixed_shape_tensor<TensorType>) {
            return fixed_subview();
        } else {
            return dynamic_subview();
        }
    }

    subview_iterator &operator++() {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            current_indices_[i] += get_subview_shape(i);
            if (current_indices_[i] < tensor_->shape()[i]) {
                break;
            }
            if (i > 0) {
                current_indices_[i] = 0;
            }
        }
        return *this;
    }

    subview_iterator operator++(int) {
        subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const subview_iterator &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const subview_iterator &other) const { return !(*this == other); }

  private:
    template <std::size_t... Is> auto fixed_subview() const {
        return tensor_->template subview<get_subview_shape(Is)...>(current_indices_[Is]...);
    }

    auto dynamic_subview() const {
        std::vector<slice> slices;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            slices.push_back({current_indices_[i], get_subview_shape(i)});
        }
        return tensor_->subview(slices);
    }

    constexpr std::size_t get_subview_shape(std::size_t i) const {
        if constexpr (std::is_array_v<SubviewShape>) {
            return subview_shape_[i];
        } else {
            return subview_shape_[i];
        }
    }
};

// Mixin class for iterable tensors
template <typename Derived, typename T> class iterable_tensor : public tensor_base<Derived, T> {
  public:
    using iterator = flat_iterator<Derived>;
    using const_iterator = flat_iterator<const Derived>;

    iterator begin() { return iterator(static_cast<Derived *>(this), 0); }
    iterator end() { return iterator(static_cast<Derived *>(this), this->size()); }

    const_iterator begin() const { return const_iterator(static_cast<const Derived *>(this), 0); }
    const_iterator end() const { return const_iterator(static_cast<const Derived *>(this), this->size()); }

    const_iterator cbegin() const { return const_iterator(static_cast<const Derived *>(this), 0); }
    const_iterator cend() const { return const_iterator(static_cast<const Derived *>(this), this->size()); }

    // Subview iteration for fixed shape tensors
    template <std::size_t... SubviewDims> auto subviews() {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");

        struct subview_range {
            Derived *tensor;
            std::array<std::size_t, sizeof...(SubviewDims)> subview_shape;

            auto begin() {
                return subview_iterator<Derived, decltype(subview_shape)>(
                    tensor, std::vector<std::size_t>(tensor->rank(), 0), subview_shape);
            }

            auto end() {
                std::vector<std::size_t> end_indices = tensor->shape();
                for (size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] -= (end_indices[i] % subview_shape[i]);
                }
                return subview_iterator<Derived, decltype(subview_shape)>(tensor, end_indices, subview_shape);
            }
        };

        return subview_range{static_cast<Derived *>(this), {SubviewDims...}};
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
                return subview_iterator<Derived, std::vector<std::size_t>>(
                    tensor, std::vector<std::size_t>(tensor->rank(), 0), subview_shape);
            }

            auto end() {
                std::vector<std::size_t> end_indices = tensor->shape();
                for (size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] -= (end_indices[i] % subview_shape[i]);
                }
                return subview_iterator<Derived, std::vector<std::size_t>>(tensor, end_indices, subview_shape);
            }
        };

        return subview_range{static_cast<Derived *>(this), subview_shape};
    }
};

} // namespace squint

#endif // SQUINT_ITERABLE_TENSOR_HPP