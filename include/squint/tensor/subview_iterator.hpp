/**
 * @file subview_iterator.hpp
 * @brief Defines the subview_iterator class for efficient iteration over tensor subviews.
 *
 * This file contains the implementation of the subview_iterator class, which provides
 * a random access iterator interface for traversing subviews of tensors. It supports
 * both fixed and dynamic tensor types, allowing for efficient iteration over subviews
 * without creating new tensor objects for each subview.
 *
 * Key features:
 * - Random access iterator capabilities
 * - Support for both fixed and dynamic tensor types
 * - Efficient subview traversal without creating intermediate tensor objects
 * - Comprehensive operator overloads for iterator manipulation
 *
 */

#ifndef SQUINT_SUBVIEW_ITERATOR_HPP
#define SQUINT_SUBVIEW_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include <iterator>

namespace squint {

/**
 * @brief A custom iterator range class.
 *
 * This class represents a range defined by two iterators. It provides a convenient
 * way to use range-based for loops with custom iterators.
 *
 * @tparam Iterator The type of iterator used to define the range.
 */
template <typename Iterator> class iterator_range {
  public:
    /**
     * @brief Construct a new iterator range object.
     *
     * @param begin The iterator pointing to the start of the range.
     * @param end The iterator pointing to the end of the range.
     */
    iterator_range(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    /**
     * @brief Get the iterator pointing to the start of the range.
     *
     * @return Iterator to the start of the range.
     */
    auto begin() const -> Iterator { return begin_; }

    /**
     * @brief Get the iterator pointing to the end of the range.
     *
     * @return Iterator to the end of the range.
     */
    auto end() const -> Iterator { return end_; }

  private:
    Iterator begin_; ///< Iterator pointing to the start of the range.
    Iterator end_;   ///< Iterator pointing to the end of the range.
};

/**
 * @brief Iterator class for tensor subviews.
 *
 * This class provides a random access iterator interface for traversing subviews of a tensor.
 * It supports both fixed and dynamic tensor types and allows efficient iteration
 * over subviews without creating new tensor objects for each subview.
 *
 * @tparam TensorType The type of the tensor being iterated.
 * @tparam SubviewShape The shape of the subview being iterated.
 */
template <typename TensorType, typename SubviewShape> class subview_iterator {
    TensorType *tensor_;                                ///< Pointer to the tensor being iterated.
    using value_type = typename TensorType::value_type; ///< Type of the tensor elements.
    using index_type = typename TensorType::index_type; ///< Type used for indexing.
    index_type current_indices_;                        ///< Current position of the iterator.
    index_type subview_shape_;                          ///< Shape of the subview.
    index_type tensor_shape_;                           ///< Shape of the entire tensor.

  public:
    /// @brief Iterator category (random access iterator).
    using iterator_category = std::random_access_iterator_tag;
    /// @brief Difference type for the iterator.
    using difference_type = std::ptrdiff_t;
    /// @brief Pointer type, const-qualified for const tensors.
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    /// @brief Reference type, const-qualified for const tensors.
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    /**
     * @brief Constructs a new subview iterator.
     * @param tensor Pointer to the tensor being iterated.
     * @param start_indices Starting indices of the subview.
     * @param subview_shape Shape of the subview.
     */
    subview_iterator(TensorType *tensor, const index_type &start_indices, const index_type &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape),
          tensor_shape_(tensor->shape()) {}

    /**
     * @brief Pre-increment operator.
     * @return Reference to the incremented iterator.
     */
    auto operator++() -> subview_iterator & {
        increment();
        return *this;
    }

    /**
     * @brief Post-increment operator.
     * @return Copy of the iterator before incrementing.
     */
    auto operator++(int) -> subview_iterator {
        auto tmp = *this;
        increment();
        return tmp;
    }

    /**
     * @brief Pre-decrement operator.
     * @return Reference to the decremented iterator.
     */
    auto operator--() -> subview_iterator & {
        decrement();
        return *this;
    }

    /**
     * @brief Post-decrement operator.
     * @return Copy of the iterator before decrementing.
     */
    auto operator--(int) -> subview_iterator {
        auto tmp = *this;
        decrement();
        return tmp;
    }

    /**
     * @brief Compound addition assignment operator.
     * @param n Number of positions to advance the iterator.
     * @return Reference to the advanced iterator.
     */
    auto operator+=(difference_type n) -> subview_iterator & {
        advance(n);
        return *this;
    }

    /**
     * @brief Compound subtraction assignment operator.
     * @param n Number of positions to move the iterator backwards.
     * @return Reference to the moved iterator.
     */
    auto operator-=(difference_type n) -> subview_iterator & {
        advance(-n);
        return *this;
    }

    /**
     * @brief Addition operator.
     * @param n Number of positions to advance the iterator.
     * @return New iterator advanced by n positions.
     */
    auto operator+(difference_type n) const -> subview_iterator {
        auto result = *this;
        result += n;
        return result;
    }

    /**
     * @brief Subtraction operator.
     * @param n Number of positions to move the iterator backwards.
     * @return New iterator moved backwards by n positions.
     */
    auto operator-(difference_type n) const -> subview_iterator {
        auto result = *this;
        result -= n;
        return result;
    }

    /**
     * @brief Difference operator between two iterators.
     * @param other Another iterator to compute the difference with.
     * @return The number of elements between the two iterators.
     */
    auto operator-(const subview_iterator &other) const -> difference_type {
        return linear_index() - other.linear_index();
    }

    /**
     * @brief Dereference operator.
     * @return A subview of the tensor at the current iterator position.
     */
    auto operator*() const {
        if constexpr (fixed_tensor<TensorType>) {
            return this->tensor_->template subview<SubviewShape>(this->get_offset());
        } else {
            std::vector<std::size_t> subview_shape = this->subview_shape_;
            while (subview_shape.back() == 1) {
                subview_shape.pop_back();
            }
            return this->tensor_->subview(subview_shape, this->get_offset());
        }
    }

    /**
     * @brief Subscript operator.
     * @param n Offset from the current position.
     * @return Subview at the offset position.
     */
    auto operator[](difference_type n) const { return *(*this + n); }

    /**
     * @brief Equality comparison operator.
     * @param other Another iterator to compare with.
     * @return true if the iterators are equal, false otherwise.
     */
    auto operator==(const subview_iterator &other) const -> bool {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    /**
     * @brief Inequality comparison operator.
     * @param other Another iterator to compare with.
     * @return true if the iterators are not equal, false otherwise.
     */
    auto operator!=(const subview_iterator &other) const -> bool { return !(*this == other); }

    /**
     * @brief Less than comparison operator.
     * @param other Another iterator to compare with.
     * @return true if this iterator is less than the other, false otherwise.
     */
    auto operator<(const subview_iterator &other) const -> bool { return linear_index() < other.linear_index(); }

    /**
     * @brief Greater than comparison operator.
     * @param other Another iterator to compare with.
     * @return true if this iterator is greater than the other, false otherwise.
     */
    auto operator>(const subview_iterator &other) const -> bool { return other < *this; }

    /**
     * @brief Less than or equal to comparison operator.
     * @param other Another iterator to compare with.
     * @return true if this iterator is less than or equal to the other, false otherwise.
     */
    auto operator<=(const subview_iterator &other) const -> bool { return !(other < *this); }

    /**
     * @brief Greater than or equal to comparison operator.
     * @param other Another iterator to compare with.
     * @return true if this iterator is greater than or equal to the other, false otherwise.
     */
    auto operator>=(const subview_iterator &other) const -> bool { return !(*this < other); }

    /**
     * @brief Addition operator for adding an integer to an iterator.
     * @param n Number of positions to advance the iterator.
     * @param it The iterator to advance.
     * @return New iterator advanced by n positions.
     */
    friend auto operator+(difference_type n, const subview_iterator &it) -> subview_iterator { return it + n; }

  private:
    /**
     * @brief Increments the iterator to the next position.
     */
    void increment() {
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i]) {
                return;
            }
            current_indices_[i] = 0;
        }
        // Set to end
        current_indices_ = tensor_shape_;
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            current_indices_[i] /= subview_shape_[i];
        }
    }

    /**
     * @brief Decrements the iterator to the previous position.
     */
    void decrement() {
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            if (current_indices_[i]-- > 0) {
                return;
            }
            current_indices_[i] = tensor_shape_[i] / subview_shape_[i] - 1;
        }
    }

    /**
     * @brief Advances the iterator by n positions.
     * @param n Number of positions to advance (can be negative).
     */
    void advance(difference_type n) {
        auto linear = linear_index() + n;
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            auto dim_size = tensor_shape_[i] / subview_shape_[i];
            current_indices_[i] = linear % dim_size;
            linear /= dim_size;
        }
    }

    /**
     * @brief Calculates the linear index of the current position.
     * @return The linear index.
     */
    [[nodiscard]] auto linear_index() const -> difference_type {
        difference_type index = 0;
        difference_type multiplier = 1;
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            index += current_indices_[i] * multiplier;
            multiplier *= tensor_shape_[i] / subview_shape_[i];
        }
        return index;
    }

    /**
     * @brief Calculates the offset for the current subview.
     * @return The index_type representing the offset of the current subview.
     */
    auto get_offset() const -> index_type {
        index_type start{};
        if constexpr (dynamic_tensor<TensorType>) {
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