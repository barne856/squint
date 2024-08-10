/**
 * @file iterable_tensor.hpp
 * @brief Defines classes for making tensors iterable and providing subview functionality.
 * 
 * This file contains the implementation of the iterator_range class and the
 * iterable_mixin class. These classes provide iteration capabilities for tensors,
 * including flat iteration over all elements and subview iteration. The file
 * supports both fixed and dynamic tensor types.
 * 
 * Key features:
 * - Custom iterator_range class for representing ranges of iterators
 * - Mixin class for adding iteration capabilities to tensor classes
 * - Support for flat iteration over all tensor elements
 * - Subview iteration for both fixed and dynamic tensor shapes
 * - Compile-time checks for subview compatibility
 * 
 */

#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/tensor/flat_iterator.hpp"
#include "squint/tensor/subview_iterator.hpp"
#include "squint/util/array_utils.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <vector>

namespace squint {

/**
 * @brief A custom iterator range class.
 * 
 * This class represents a range defined by two iterators. It provides a convenient
 * way to use range-based for loops with custom iterators.
 * 
 * @tparam Iterator The type of iterator used to define the range.
 */
template <typename Iterator>
class iterator_range {
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
    Iterator begin_;  ///< Iterator pointing to the start of the range.
    Iterator end_;    ///< Iterator pointing to the end of the range.
};

/**
 * @brief Mixin class for adding iteration capabilities to tensor classes.
 * 
 * This class provides methods for flat iteration over all tensor elements and
 * subview iteration. It supports both fixed and dynamic tensor types.
 * 
 * @tparam Derived The derived tensor class that inherits from this mixin.
 */
template <typename Derived>
class iterable_mixin {
  public:
    using iterator = flat_iterator<Derived>;              ///< Type alias for the flat iterator.
    using const_iterator = flat_iterator<const Derived>;  ///< Type alias for the const flat iterator.

    /**
     * @brief Get an iterator to the beginning of the tensor.
     * 
     * @return iterator Pointing to the first element of the tensor.
     */
    auto begin() {
        typename Derived::index_type start_indices{};
        if constexpr (dynamic_tensor<Derived>) {
            start_indices.resize(static_cast<Derived *>(this)->rank(), 0);
        }
        return iterator(static_cast<Derived *>(this), start_indices);
    }

    /**
     * @brief Get an iterator to the end of the tensor.
     * 
     * @return iterator Pointing one past the last element of the tensor.
     */
    auto end() { return iterator(static_cast<Derived *>(this), static_cast<Derived *>(this)->shape()); }

    /**
     * @brief Get a const iterator to the beginning of the tensor.
     * 
     * @return const_iterator Pointing to the first element of the tensor.
     */
    auto begin() const {
        typename Derived::index_type start_indices{};
        if constexpr (dynamic_tensor<Derived>) {
            start_indices.resize(static_cast<const Derived *>(this)->rank(), 0);
        }
        return const_iterator(static_cast<const Derived *>(this), start_indices);
    }

    /**
     * @brief Get a const iterator to the end of the tensor.
     * 
     * @return const_iterator Pointing one past the last element of the tensor.
     */
    auto end() const {
        return const_iterator(static_cast<const Derived *>(this), static_cast<Derived *>(this)->shape());
    }

    /**
     * @brief Get a const iterator to the beginning of the tensor.
     * 
     * @return const_iterator Pointing to the first element of the tensor.
     */
    auto cbegin() const { return begin(); }

    /**
     * @brief Get a const iterator to the end of the tensor.
     * 
     * @return const_iterator Pointing one past the last element of the tensor.
     */
    auto cend() const { return end(); }

    /**
     * @brief Get an iterator range for subviews of a fixed shape tensor.
     * 
     * @tparam SubviewShape The shape of the subviews.
     * @return iterator_range<subview_iterator> Range of subview iterators.
     */
    template <typename SubviewShape>
    auto subviews()
        requires fixed_tensor<Derived> && fixed_shape<SubviewShape>
    {
        auto derived = static_cast<Derived *>(this);
        static_assert(SubviewShape::size() == Derived::shape_type::size(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewShape>(),
                      "Subview dimensions must evenly divide tensor dimensions");
        auto begin = subview_iterator<Derived, SubviewShape>(derived, std::array<std::size_t, SubviewShape::size()>{});
        auto end = subview_iterator<Derived, SubviewShape>(derived, [] {
            constexpr auto end_indices = make_array(typename Derived::shape_type{});
            constexpr auto subview_shape = make_array(SubviewShape{});
            std::array<std::size_t, SubviewShape::size()> result;
            std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), result.begin(),
                           std::divides<>());
            return result;
        }());
        return iterator_range(begin, end);
    }

    /**
     * @brief Get an iterator range for subviews of a fixed shape tensor.
     * 
     * @tparam Dims Variadic template parameter pack for subview dimensions.
     * @return iterator_range<subview_iterator> Range of subview iterators.
     */
    template <std::size_t... Dims>
    auto subviews()
        requires fixed_tensor<Derived>
    {
        return subviews<std::index_sequence<Dims...>>();
    }

    /**
     * @brief Get a const iterator range for subviews of a fixed shape tensor.
     * 
     * @tparam SubviewShape The shape of the subviews.
     * @return iterator_range<const_subview_iterator> Range of const subview iterators.
     */
    template <typename SubviewShape>
    auto subviews() const
        requires fixed_tensor<Derived> && fixed_shape<SubviewShape>
    {
        auto derived = static_cast<const Derived *>(this);
        static_assert(SubviewShape::size() == Derived::shape_type::size(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewShape>(),
                      "Subview dimensions must evenly divide tensor dimensions");
        auto begin =
            subview_iterator<const Derived, SubviewShape>(derived, std::array<std::size_t, SubviewShape::size()>{});
        auto end = subview_iterator<const Derived, SubviewShape>(derived, [] {
            constexpr auto end_indices = make_array(typename Derived::shape_type{});
            constexpr auto subview_shape = make_array(SubviewShape{});
            std::array<std::size_t, SubviewShape::size()> result;
            std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), result.begin(),
                           std::divides<>());
            return result;
        }());
        return iterator_range(begin, end);
    }

    /**
     * @brief Get a const iterator range for subviews of a fixed shape tensor.
     * 
     * @tparam Dims Variadic template parameter pack for subview dimensions.
     * @return iterator_range<const_subview_iterator> Range of const subview iterators.
     */
    template <std::size_t... Dims>
    auto subviews() const
        requires fixed_tensor<Derived>
    {
        return subviews<std::index_sequence<Dims...>>();
    }

    /**
     * @brief Get an iterator range for subviews of a dynamic shape tensor.
     * 
     * @param subview_shape Vector specifying the shape of the subviews.
     * @return iterator_range<subview_iterator> Range of subview iterators.
     * @throws std::invalid_argument If subview dimensions are invalid.
     */
    auto subviews(const std::vector<std::size_t> &subview_shape)
        requires dynamic_tensor<Derived>
    {
        auto derived = static_cast<Derived *>(this);
        if constexpr (derived->error_checking() == error_checking::enabled) {
            if (subview_shape.size() != derived->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                derived->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }
        auto begin = subview_iterator<Derived, std::vector<std::size_t>>(
            derived, std::vector<std::size_t>(derived->rank(), 0), subview_shape);
        auto end = subview_iterator<Derived, std::vector<std::size_t>>(
            derived,
            [this, &subview_shape] {
                auto end_indices = derived->shape();
                std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                               std::divides<>());
                return end_indices;
            }(),
            subview_shape);
        return iterator_range(begin, end);
    }

    /**
     * @brief Get a const iterator range for subviews of a dynamic shape tensor.
     * 
     * @param subview_shape Vector specifying the shape of the subviews.
     * @return iterator_range<const_subview_iterator> Range of const subview iterators.
     * @throws std::invalid_argument If subview dimensions are invalid.
     */
    auto subviews(const std::vector<std::size_t> &subview_shape) const
        requires dynamic_tensor<Derived>
    {
        auto derived = static_cast<const Derived *>(this);
        if constexpr (derived->error_checking() == error_checking::enabled) {
            if (subview_shape.size() != derived->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                derived->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }
        auto begin = subview_iterator<const Derived, std::vector<std::size_t>>(
            derived, std::vector<std::size_t>(derived->rank(), 0), subview_shape);
        auto end = subview_iterator<const Derived, std::vector<std::size_t>>(
            derived,
            [this, &subview_shape] {
                auto end_indices = derived->shape();
                std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                               std::divides<>());
                return end_indices;
            }(),
            subview_shape);
        return iterator_range(begin, end);
    }
};

} // namespace squint

#endif // SQUINT_ITERABLE_TENSOR_HPP