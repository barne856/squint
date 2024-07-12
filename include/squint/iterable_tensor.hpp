#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/tensor_base.hpp"
#include <array>
#include <iterator>
#include <numeric>
#include <vector>

namespace squint {

// Flat iterator for tensors
template <typename TensorType> class flat_iterator {
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> shape_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    flat_iterator(TensorType *tensor, const std::vector<std::size_t> &indices)
        : tensor_(tensor), current_indices_(indices), shape_(tensor->shape()) {}

    reference operator*() const { return tensor_->at_impl(current_indices_); }

    pointer operator->() const { return &(operator*()); }

    flat_iterator &operator++() {
        for (int i = 0; i < current_indices_.size(); ++i) {
            if (++current_indices_[i] < shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        // If we've reached here, we've gone past the end
        current_indices_ = shape_; // Set to end
        return *this;
    }

    flat_iterator operator++(int) {
        flat_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    flat_iterator &operator--() {
        for (int i = 0; i < current_indices_.size(); ++i) {
            if (current_indices_[i]-- > 0) {
                return *this;
            }
            current_indices_[i] = shape_[i] - 1;
        }
        // If we've reached here, we've gone before the beginning
        std::fill(current_indices_.begin(), current_indices_.end(), 0);
        return *this;
    }

    flat_iterator operator--(int) {
        flat_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    flat_iterator &operator+=(difference_type n) {
        while (n > 0) {
            ++(*this);
            --n;
        }
        while (n < 0) {
            --(*this);
            ++n;
        }
        return *this;
    }

    flat_iterator &operator-=(difference_type n) { return *this += -n; }

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
        // This is an approximation and might be slow for large differences
        difference_type diff = 0;
        flat_iterator temp = *this;
        while (temp != other) {
            if (temp < other) {
                ++temp;
                ++diff;
            } else {
                --temp;
                --diff;
            }
        }
        return diff;
    }

    reference operator[](difference_type n) const { return *(*this + n); }

    bool operator==(const flat_iterator &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const flat_iterator &other) const { return !(*this == other); }
    bool operator<(const flat_iterator &other) const {
        return std::lexicographical_compare(current_indices_.begin(), current_indices_.end(),
                                            other.current_indices_.begin(), other.current_indices_.end());
    }
    bool operator>(const flat_iterator &other) const { return other < *this; }
    bool operator<=(const flat_iterator &other) const { return !(other < *this); }
    bool operator>=(const flat_iterator &other) const { return !(*this < other); }

    friend flat_iterator operator+(difference_type n, const flat_iterator &it) { return it + n; }
};

template <typename TensorType> class subview_iterator_base {
  protected:
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    TensorType *tensor_;
    std::vector<std::size_t> current_indices_;
    std::vector<std::size_t> subview_shape_;
    std::vector<std::size_t> tensor_shape_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;

    subview_iterator_base(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                          const std::vector<std::size_t> &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape),
          tensor_shape_(tensor->shape()) {}

    subview_iterator_base &operator++() {
        for (int i = 0; i < current_indices_.size(); ++i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        // If we've reached here, we've gone past the end
        // Set to end
        current_indices_ = tensor_shape_;
        size_t i = 0;
        for (auto &index : current_indices_) {
            index /= subview_shape_[i++];
        }
        return *this;
    }

    bool operator==(const subview_iterator_base &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const subview_iterator_base &other) const { return !(*this == other); }

  protected:
    std::vector<slice> get_slices() const {
        std::vector<slice> slices;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            slices.push_back({current_indices_[i] * subview_shape_[i], subview_shape_[i]});
        }
        return slices;
    }
};

// Fixed subview iterator for tensors
template <typename TensorType, std::size_t... SubviewDims>
class fixed_subview_iterator : public subview_iterator_base<TensorType> {
    using Base = subview_iterator_base<TensorType>;

  public:
    fixed_subview_iterator(TensorType *tensor, const std::array<std::size_t, sizeof...(SubviewDims)> &start_indices)
        : Base(tensor, std::vector<std::size_t>(start_indices.begin(), start_indices.end()),
               std::vector<std::size_t>{SubviewDims...}) {}

    auto operator*() const { return make_subview(std::make_index_sequence<sizeof...(SubviewDims)>{}); }

    fixed_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    fixed_subview_iterator operator++(int) {
        fixed_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

  private:
    template <std::size_t... Is> auto make_subview(std::index_sequence<Is...> /*unused*/) const {
        return this->tensor_->template subview<SubviewDims...>(
            slice{this->current_indices_[Is] * this->subview_shape_[Is], this->subview_shape_[Is]}...);
    }
};

// Const fixed subview iterator for tensors
template <typename TensorType, std::size_t... SubviewDims>
class const_fixed_subview_iterator : public subview_iterator_base<const TensorType> {
    using Base = subview_iterator_base<const TensorType>;

  public:
    const_fixed_subview_iterator(const TensorType *tensor,
                                 const std::array<std::size_t, sizeof...(SubviewDims)> &start_indices)
        : Base(tensor, std::vector<std::size_t>(start_indices.begin(), start_indices.end()),
               std::vector<std::size_t>{SubviewDims...}) {}

    auto operator*() const { return make_subview(std::make_index_sequence<sizeof...(SubviewDims)>{}); }

    const_fixed_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    const_fixed_subview_iterator operator++(int) {
        const_fixed_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

  private:
    template <std::size_t... Is> auto make_subview(std::index_sequence<Is...> /*unused*/) const {
        return this->tensor_->template subview<SubviewDims...>(
            slice{this->current_indices_[Is] * this->subview_shape_[Is], this->subview_shape_[Is]}...);
    }
};

// Dynamic subview iterator for tensors
template <typename TensorType> class dynamic_subview_iterator : public subview_iterator_base<TensorType> {
    using Base = subview_iterator_base<TensorType>;

  public:
    dynamic_subview_iterator(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                             const std::vector<std::size_t> &subview_shape)
        : Base(tensor, start_indices, subview_shape) {}

    auto operator*() const { return this->tensor_->subview(this->get_slices()); }

    dynamic_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    dynamic_subview_iterator operator++(int) {
        dynamic_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
};

// Const dynamic subview iterator for tensors
template <typename TensorType> class const_dynamic_subview_iterator : public subview_iterator_base<const TensorType> {
    using Base = subview_iterator_base<const TensorType>;

  public:
    const_dynamic_subview_iterator(const TensorType *tensor, const std::vector<std::size_t> &start_indices,
                                   const std::vector<std::size_t> &subview_shape)
        : Base(tensor, start_indices, subview_shape) {}

    auto operator*() const { return this->tensor_->subview(this->get_slices()); }

    const_dynamic_subview_iterator &operator++() {
        Base::operator++();
        return *this;
    }

    const_dynamic_subview_iterator operator++(int) {
        const_dynamic_subview_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
};

// Mixin class for iterable tensors
template <typename Derived, typename T, error_checking ErrorChecking>
class iterable_tensor : public tensor_base<Derived, T, ErrorChecking> {
  public:
    using iterator = flat_iterator<Derived>;
    using const_iterator = flat_iterator<const Derived>;

    iterator begin() { return iterator(static_cast<Derived *>(this), std::vector<std::size_t>(this->rank(), 0)); }

    iterator end() { return iterator(static_cast<Derived *>(this), this->shape()); }

    const_iterator begin() const {
        return const_iterator(static_cast<const Derived *>(this), std::vector<std::size_t>(this->rank(), 0));
    }

    const_iterator end() const { return const_iterator(static_cast<const Derived *>(this), this->shape()); }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // Helper function to check if tensor dimensions are divisible by subview dimensions
    template <std::size_t... SubviewDims, std::size_t... Is>
    static constexpr bool dimensions_divisible_helper(const std::array<std::size_t, sizeof...(Is)> &tensor_shape,
                                                      std::index_sequence<Is...> /*unused*/) {
        return ((tensor_shape[Is] % SubviewDims == 0) && ...);
    }

    // Subview iteration for fixed shape tensors
    template <std::size_t... SubviewDims> auto subviews() {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");

        // Get the shape of the tensor
        constexpr auto tensor_shape = Derived::constexpr_shape();

        // Check if each dimension of the tensor is evenly divisible by the subview dimensions
        static_assert(dimensions_divisible_helper<SubviewDims...>(tensor_shape,
                                                                  std::make_index_sequence<sizeof...(SubviewDims)>{}),
                      "Subview dimensions must evenly divide tensor dimensions");

        struct subview_range {
            Derived *tensor;

            auto begin() {
                return fixed_subview_iterator<Derived, SubviewDims...>(
                    tensor, std::array<std::size_t, sizeof...(SubviewDims)>());
            }
            auto end() {
                auto end_indices = Derived::constexpr_shape();
                // divide shape by subview dims to get the end indices
                size_t i = 0;
                for (auto &index : end_indices) {
                    index /= std::array{SubviewDims...}[i++];
                }
                return fixed_subview_iterator<Derived, SubviewDims...>(tensor, end_indices);
            }
        };

        return subview_range{static_cast<Derived *>(this)};
    }

    // Const subview iteration for fixed shape tensors
    template <std::size_t... SubviewDims> auto subviews() const {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");

        // Get the shape of the tensor
        constexpr auto tensor_shape = Derived::constexpr_shape();

        // Check if each dimension of the tensor is evenly divisible by the subview dimensions
        static_assert(dimensions_divisible_helper<SubviewDims...>(tensor_shape,
                                                                  std::make_index_sequence<sizeof...(SubviewDims)>{}),
                      "Subview dimensions must evenly divide tensor dimensions");

        struct const_subview_range {
            const Derived *tensor;

            auto begin() const {
                return const_fixed_subview_iterator<Derived, SubviewDims...>(
                    tensor, std::array<std::size_t, sizeof...(SubviewDims)>());
            }

            auto end() const {
                auto end_indices = Derived::constexpr_shape();
                // divide shape by subview dims to get the end indices
                size_t i = 0;
                for (auto &index : end_indices) {
                    index /= std::array{SubviewDims...}[i++];
                }
                return const_fixed_subview_iterator<Derived, SubviewDims...>(tensor, end_indices);
            }
        };

        return const_subview_range{static_cast<const Derived *>(this)};
    }

    // Subview iteration for dynamic shape tensors
    auto subviews(const std::vector<std::size_t> &subview_shape) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (subview_shape.size() != this->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            // New dimensions must evenly divide old dimensions
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                this->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }

        struct subview_range {
            Derived *tensor;
            std::vector<std::size_t> subview_shape;

            auto begin() {
                return dynamic_subview_iterator<Derived>(tensor, std::vector<std::size_t>(tensor->rank(), 0),
                                                         subview_shape);
            }

            auto end() {
                auto end_indices = tensor->shape();
                for (std::size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] /= subview_shape[i];
                }
                return dynamic_subview_iterator<Derived>(tensor, end_indices, subview_shape);
            }
        };

        return subview_range{static_cast<Derived *>(this), subview_shape};
    }

    // Const subview iteration for dynamic shape tensors
    auto subviews(const std::vector<std::size_t> &subview_shape) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (subview_shape.size() != this->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            // New dimensions must evenly divide old dimensions
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                this->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }

        struct const_subview_range {
            const Derived *tensor;
            std::vector<std::size_t> subview_shape;

            auto begin() const {
                return const_dynamic_subview_iterator<Derived>(tensor, std::vector<std::size_t>(tensor->rank(), 0),
                                                               subview_shape);
            }

            auto end() const {
                auto end_indices = tensor->shape();
                for (std::size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] /= subview_shape[i];
                }
                return const_dynamic_subview_iterator<Derived>(tensor, end_indices, subview_shape);
            }
        };

        return const_subview_range{static_cast<const Derived *>(this), subview_shape};
    }
};

} // namespace squint

// Iterator traits specialization
template <typename TensorType> struct std::iterator_traits<squint::flat_iterator<TensorType>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::remove_const<typename TensorType::value_type>::type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<std::is_const_v<TensorType>, const value_type *, value_type *>;
    using reference = std::conditional_t<std::is_const_v<TensorType>, const value_type &, value_type &>;
};

#endif // SQUINT_ITERABLE_TENSOR_HPP