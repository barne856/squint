/**
 * @file subview_iterator.hpp
 * @brief Defines the subview_iterator class for iterating over subviews of tensors.
 *
 * This file contains the implementation of the subview_iterator class, which allows
 * efficient iteration over subviews of tensors. It supports both fixed and dynamic
 * tensor types and provides a forward iterator interface.
 *
 */

#ifndef SQUINT_SUBVIEW_ITERATOR_HPP
#define SQUINT_SUBVIEW_ITERATOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/util/array_utils.hpp"

namespace squint {

/**
 * @brief Iterator class for tensor subviews.
 *
 * This class provides an iterator interface for traversing subviews of a tensor.
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
    /// @brief Iterator category (forward iterator).
    using iterator_category = std::forward_iterator_tag;
    /// @brief Difference type for the iterator.
    using difference_type = std::ptrdiff_t;
    /// @brief Pointer type, const-qualified for const tensors.
    using pointer = std::conditional_t<const_tensor<TensorType>, const value_type *, value_type *>;
    /// @brief Reference type, const-qualified for const tensors.
    using reference = std::conditional_t<const_tensor<TensorType>, const value_type &, value_type &>;

    /**
     * @brief Constructor
     *
     * @param tensor Pointer to the tensor being iterated.
     * @param start_indices Starting indices of the subview.
     * @param subview_shape Shape of the subview.
     */
    subview_iterator(TensorType *tensor, const index_type &start_indices, const index_type &subview_shape)
        : tensor_(tensor), current_indices_(start_indices), subview_shape_(subview_shape),
          tensor_shape_(tensor->shape()) {}

    /**
     * @brief Pre-increment operator.
     *
     * Advances the iterator to the next subview position.
     *
     * @return Reference to the updated iterator.
     */
    auto operator++() -> subview_iterator & {
        for (int i = current_indices_.size() - 1; i >= 0; --i) {
            if (++current_indices_[i] < tensor_shape_[i] / subview_shape_[i]) {
                return *this;
            }
            current_indices_[i] = 0;
        }
        // Set to end
        current_indices_ = tensor_shape_;
        for (size_t i = 0; i < current_indices_.size(); ++i) {
            current_indices_[i] /= subview_shape_[i];
        }
        return *this;
    }

    /**
     * @brief Post-increment operator.
     *
     * Creates a copy of the iterator, then advances it.
     *
     * @return Copy of the iterator before incrementing.
     */
    auto operator++(int) -> subview_iterator {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Dereference operator.
     *
     * @return A subview of the tensor at the current iterator position.
     */
    auto operator*() const {
        if constexpr (fixed_tensor<TensorType>) {
            return this->tensor_->template subview<SubviewShape>(this->get_offset());
        } else {
            // remove trailing 1s from the subview shape
            std::vector<std::size_t> subview_shape = this->subview_shape_;
            while (subview_shape.back() == 1) {
                subview_shape.pop_back();
            }
            return this->tensor_->subview(subview_shape, this->get_offset());
        }
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param other Another subview_iterator to compare with.
     * @return true if the iterators are equal, false otherwise.
     */
    auto operator==(const subview_iterator &other) const -> bool {
        return tensor_ == other.tensor_ && current_indices_ == other.current_indices_;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param other Another subview_iterator to compare with.
     * @return true if the iterators are not equal, false otherwise.
     */
    auto operator!=(const subview_iterator &other) const -> bool { return !(*this == other); }

  private:
    /**
     * @brief Calculates the offset for the current subview.
     *
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