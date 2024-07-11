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
    std::vector<std::size_t> strides_;
    std::size_t flat_index_;

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
        flat_index_ = flatten_index(indices);
    }

    reference operator*() const { return tensor_->data()[flat_index_]; }

    pointer operator->() const { return &(operator*()); }

    flat_iterator &operator++() {
        ++flat_index_;
        if (flat_index_ < tensor_->size()) {
            unflatten_index(flat_index_, current_indices_);
        }
        return *this;
    }

    flat_iterator operator++(int) {
        flat_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    flat_iterator &operator--() {
        if (flat_index_ > 0) {
            --flat_index_;
            unflatten_index(flat_index_, current_indices_);
        }
        return *this;
    }

    flat_iterator operator--(int) {
        flat_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    flat_iterator &operator+=(difference_type n) {
        flat_index_ = std::min(flat_index_ + n, tensor_->size());
        if (flat_index_ < tensor_->size()) {
            unflatten_index(flat_index_, current_indices_);
        }
        return *this;
    }

    flat_iterator &operator-=(difference_type n) {
        if (n <= flat_index_) {
            flat_index_ -= n;
            unflatten_index(flat_index_, current_indices_);
        } else {
            flat_index_ = 0;
            std::fill(current_indices_.begin(), current_indices_.end(), 0);
        }
        return *this;
    }

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
        return static_cast<difference_type>(flat_index_) - static_cast<difference_type>(other.flat_index_);
    }

    reference operator[](difference_type n) const { return tensor_->data()[flat_index_ + n]; }

    bool operator==(const flat_iterator &other) const {
        return tensor_ == other.tensor_ && flat_index_ == other.flat_index_;
    }

    bool operator!=(const flat_iterator &other) const { return !(*this == other); }
    bool operator<(const flat_iterator &other) const { return flat_index_ < other.flat_index_; }
    bool operator>(const flat_iterator &other) const { return other < *this; }
    bool operator<=(const flat_iterator &other) const { return !(other < *this); }
    bool operator>=(const flat_iterator &other) const { return !(*this < other); }

    friend flat_iterator operator+(difference_type n, const flat_iterator &it) { return it + n; }

  private:
    std::size_t flatten_index(const std::vector<std::size_t> &indices) const {
        std::size_t flat_index = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            flat_index += indices[i] * strides_[i];
        }
        return flat_index;
    }

    void unflatten_index(std::size_t flat_index, std::vector<std::size_t> &indices) const {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            indices[i] = flat_index / strides_[i];
            flat_index %= strides_[i];
        }
    }
};

template <typename TensorType> class subview_iterator_base {
  protected:
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

    subview_iterator_base(TensorType *tensor, const std::vector<std::size_t> &start_indices,
                          const std::vector<std::size_t> &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape) {
        if constexpr (fixed_shape_tensor<TensorType>) {
            constexpr auto tensor_strides = TensorType::constexpr_strides();
            strides_.assign(tensor_strides.begin(), tensor_strides.end());
        } else {
            strides_ = tensor->strides();
        }
    }

    subview_iterator_base &operator++() {
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

    bool operator==(const subview_iterator_base &other) const {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    bool operator!=(const subview_iterator_base &other) const { return !(*this == other); }

  protected:
    std::vector<slice> get_slices() const {
        std::vector<slice> slices;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            slices.push_back({current_indices_[i], subview_shape_[i]});
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
            slice{this->current_indices_[Is], this->subview_shape_[Is]}...);
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
            slice{this->current_indices_[Is], this->subview_shape_[Is]}...);
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

    iterator end() {
        return iterator(static_cast<Derived *>(this), std::vector<std::size_t>(this->rank(), 0)) + this->size();
    }

    const_iterator begin() const {
        return const_iterator(static_cast<const Derived *>(this), std::vector<std::size_t>(this->rank(), 0));
    }

    const_iterator end() const {
        return const_iterator(static_cast<const Derived *>(this), std::vector<std::size_t>(this->rank(), 0)) +
               this->size();
    }

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
                std::array<std::size_t, sizeof...(SubviewDims)> end_indices;
                auto shape = tensor->shape();
                std::size_t i = 0;
                ((end_indices[i] = shape[i] - (shape[i] % SubviewDims), ++i), ...);
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
                std::array<std::size_t, sizeof...(SubviewDims)> end_indices;
                auto shape = tensor->shape();
                std::size_t i = 0;
                ((end_indices[i] = shape[i] - (shape[i] % SubviewDims), ++i), ...);
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
                std::vector<std::size_t> end_indices = tensor->shape();
                for (size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] -= (end_indices[i] % subview_shape[i]);
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
                std::vector<std::size_t> end_indices = tensor->shape();
                for (size_t i = 0; i < end_indices.size(); ++i) {
                    end_indices[i] -= (end_indices[i] % subview_shape[i]);
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