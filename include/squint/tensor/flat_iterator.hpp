/**
 * @file flat_iterator.hpp
 * @brief Defines the flat_iterator class for linear traversal of multi-dimensional tensors.
 *
 * This file contains the implementation of the flat_iterator class, which provides
 * a way to iterate over all elements of a tensor in a linear fashion, regardless of
 * its dimensionality. The iterator supports both fixed and dynamic tensor types and
 * satisfies the requirements of a random access iterator.
 *
 * Key features:
 * - Linear traversal of multi-dimensional tensors
 * - Support for both fixed and dynamic tensor types
 * - Random access iterator capabilities
 * - Arithmetic operations for iterator manipulation
 * - Comparison operations between iterators
 *
 */

#ifndef SQUINT_FLAT_ITERATOR_HPP
#define SQUINT_FLAT_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/util/sequence_utils.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

namespace squint {

/**
 * @brief A flat iterator for linear traversal of tensor elements.
 *
 * This class provides a way to iterate over all elements of a tensor in a linear
 * fashion, regardless of its dimensionality. It satisfies the requirements of a
 * random access iterator.
 *
 * @tparam TensorType The type of the tensor being iterated.
 */
template <typename TensorType> class flat_iterator {
  public:
    using index_type = typename TensorType::index_type; ///< Type used for indexing.
    using value_type = typename TensorType::value_type; ///< Type of the tensor elements.
    /// @brief Iterator category (random access iterator).
    using iterator_category = std::random_access_iterator_tag;
    /// @brief Difference type for the iterator.
    using difference_type = std::ptrdiff_t;
    /// @brief Pointer type, const-qualified for const tensors.
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    /// @brief Reference type, const-qualified for const tensors.
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    /**
     * @brief Construct a new flat iterator object.
     *
     * @param tensor Pointer to the tensor being iterated.
     * @param indices Starting indices of the iterator.
     */
    flat_iterator(TensorType *tensor, const index_type &indices) : tensor_(tensor), current_indices_(indices) {
        if constexpr (fixed_tensor<TensorType>) {
            shape_ = make_array(typename TensorType::shape_type{});
            strides_ = make_array(typename TensorType::strides_type{});
        } else {
            shape_ = tensor->shape();
            strides_ = tensor->strides();
        }
    }

    /**
     * @brief Dereference operator.
     *
     * @return reference A reference to the current element.
     */
    auto operator*() const -> reference {
        return tensor_
            ->data()[std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL)];
    }

    /**
     * @brief Arrow operator.
     *
     * @return pointer A pointer to the current element.
     */
    auto operator->() const -> pointer { return &(operator*()); }

    /**
     * @brief Pre-increment operator.
     *
     * @return flat_iterator& Reference to the incremented iterator.
     */
    auto operator++() -> flat_iterator & {
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            if (++current_indices_[i] < shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        current_indices_ = shape_; // Set to end
        return *this;
    }

    /**
     * @brief Post-increment operator.
     *
     * @return flat_iterator Copy of the iterator before incrementing.
     */
    auto operator++(int) -> flat_iterator {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Pre-decrement operator.
     *
     * @return flat_iterator& Reference to the decremented iterator.
     */
    auto operator--() -> flat_iterator & {
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            if (current_indices_[i]-- > 0) {
                return *this;
            }
            current_indices_[i] = shape_[i] - 1;
        }
        std::fill(current_indices_.begin(), current_indices_.end(), 0); // Set to begin
        return *this;
    }

    /**
     * @brief Post-decrement operator.
     *
     * @return flat_iterator Copy of the iterator before decrementing.
     */
    auto operator--(int) -> flat_iterator {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    /**
     * @brief Compound addition assignment operator.
     *
     * @param n Number of positions to advance the iterator.
     * @return flat_iterator& Reference to the advanced iterator.
     */
    auto operator+=(difference_type n) -> flat_iterator & {
        std::size_t total_offset =
            std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0ULL) + n;
        for (std::size_t i = 0; i < current_indices_.size(); ++i) {
            current_indices_[i] = total_offset / strides_[i];
            total_offset %= strides_[i];
        }
        return *this;
    }

    /**
     * @brief Compound subtraction assignment operator.
     *
     * @param n Number of positions to move the iterator backwards.
     * @return flat_iterator& Reference to the moved iterator.
     */
    auto operator-=(difference_type n) -> flat_iterator & { return *this += -n; }

    /**
     * @brief Addition operator.
     *
     * @param n Number of positions to advance the iterator.
     * @return flat_iterator New iterator advanced by n positions.
     */
    auto operator+(difference_type n) const -> flat_iterator {
        auto result = *this;
        result += n;
        return result;
    }

    /**
     * @brief Subtraction operator.
     *
     * @param n Number of positions to move the iterator backwards.
     * @return flat_iterator New iterator moved backwards by n positions.
     */
    auto operator-(difference_type n) const -> flat_iterator {
        auto result = *this;
        result -= n;
        return result;
    }

    /**
     * @brief Difference operator between two iterators.
     *
     * @param other Another iterator to compute the difference with.
     * @return difference_type The number of elements between the two iterators.
     */
    auto operator-(const flat_iterator &other) const -> difference_type {
        return std::inner_product(current_indices_.begin(), current_indices_.end(), strides_.begin(), 0LL) -
               std::inner_product(other.current_indices_.begin(), other.current_indices_.end(), other.strides_.begin(),
                                  0LL);
    }

    /**
     * @brief Subscript operator.
     *
     * @param n Offset from the current position.
     * @return reference Reference to the element at the offset position.
     */
    auto operator[](difference_type n) const -> reference { return *(*this + n); }

    /**
     * @brief Equality comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if the iterators are equal, false otherwise.
     */
    auto operator==(const flat_iterator &other) const -> bool {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if the iterators are not equal, false otherwise.
     */
    auto operator!=(const flat_iterator &other) const -> bool { return !(*this == other); }

    /**
     * @brief Less than comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if this iterator is less than the other, false otherwise.
     */
    auto operator<(const flat_iterator &other) const -> bool {
        return std::lexicographical_compare(current_indices_.begin(), current_indices_.end(),
                                            other.current_indices_.begin(), other.current_indices_.end());
    }

    /**
     * @brief Greater than comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if this iterator is greater than the other, false otherwise.
     */
    auto operator>(const flat_iterator &other) const -> bool { return other < *this; }

    /**
     * @brief Less than or equal to comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if this iterator is less than or equal to the other, false otherwise.
     */
    auto operator<=(const flat_iterator &other) const -> bool { return !(other < *this); }

    /**
     * @brief Greater than or equal to comparison operator.
     *
     * @param other Another iterator to compare with.
     * @return true if this iterator is greater than or equal to the other, false otherwise.
     */
    auto operator>=(const flat_iterator &other) const -> bool { return !(*this < other); }

    /**
     * @brief Friend function for addition of an integer and an iterator.
     *
     * @param n Number of positions to advance the iterator.
     * @param it The iterator to advance.
     * @return flat_iterator New iterator advanced by n positions.
     */
    friend auto operator+(difference_type n, const flat_iterator &it) -> flat_iterator { return it + n; }

  private:
    TensorType *tensor_;         ///< Pointer to the tensor being iterated.
    index_type current_indices_; ///< Current position of the iterator.
    index_type shape_;           ///< Shape of the tensor.
    index_type strides_;         ///< Strides of the tensor.
};

} // namespace squint

#endif // SQUINT_FLAT_ITERATOR_HPP