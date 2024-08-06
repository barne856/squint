#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/tensor/flat_iterator.hpp"
#include "squint/tensor/subview_iterator.hpp"
#include "squint/tensor/tensor_base.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <vector>

namespace squint {

// Mixin class for iterable tensors
template <typename Derived, typename T, typename Shape, layout Layout, error_checking ErrorChecking>
class iterable_tensor : public tensor_base<Derived, T, Shape, Layout, ErrorChecking> {
  public:
    using iterator = flat_iterator<Derived>;
    using const_iterator = flat_iterator<const Derived>;

    auto begin() { return iterator(static_cast<Derived *>(this), std::vector<std::size_t>(this->rank(), 0)); }
    auto end() { return iterator(static_cast<Derived *>(this), this->shape()); }
    auto begin() const {
        return const_iterator(static_cast<const Derived *>(this), std::vector<std::size_t>(this->rank(), 0));
    }
    auto end() const { return const_iterator(static_cast<const Derived *>(this), this->shape()); }
    auto cbegin() const { return begin(); }
    auto cend() const { return end(); }

    // Subview iteration for fixed shape tensors
    template <std::size_t... SubviewDims>
    auto subviews()
        requires fixed_shape<Derived>
    {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewDims...>(),
                      "Subview dimensions must evenly divide tensor dimensions");

        auto derived = static_cast<Derived *>(this);

        return std::ranges::subrange(
            fixed_subview_iterator<Derived, SubviewDims...>(derived, std::array<std::size_t, sizeof...(SubviewDims)>{}),
            fixed_subview_iterator<Derived, SubviewDims...>(derived, [] {
                auto end_indices = derived->shape();
                std::transform(end_indices.begin(), end_indices.end(), std::array{SubviewDims...}.begin(),
                               end_indices.begin(), std::divides<>());
                return end_indices;
            }()));
    }

    template <std::size_t... SubviewDims>
    auto subviews() const
        requires fixed_shape<Derived>
    {
        static_assert(sizeof...(SubviewDims) == Derived::rank(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewDims...>(),
                      "Subview dimensions must evenly divide tensor dimensions");

        auto derived = static_cast<const Derived *>(this);

        return std::ranges::subrange(fixed_subview_iterator<const Derived, SubviewDims...>(
                                         derived, std::array<std::size_t, sizeof...(SubviewDims)>{}),
                                     fixed_subview_iterator<const Derived, SubviewDims...>(derived, [] {
                                         auto end_indices = derived->shape();
                                         std::transform(end_indices.begin(), end_indices.end(),
                                                        std::array{SubviewDims...}.begin(), end_indices.begin(),
                                                        std::divides<>());
                                         return end_indices;
                                     }()));
    }

    // Subview iteration for dynamic shape tensors
    auto subviews(const std::vector<std::size_t> &subview_shape)
        requires dynamic_shape<Derived>
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (subview_shape.size() != this->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                this->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }

        return std::ranges::subrange(
            dynamic_subview_iterator<Derived>(static_cast<Derived *>(this), std::vector<std::size_t>(this->rank(), 0),
                                              subview_shape),
            dynamic_subview_iterator<Derived>(
                static_cast<Derived *>(this),
                [this, &subview_shape] {
                    auto end_indices = this->shape();
                    std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                                   std::divides<>());
                    return end_indices;
                }(),
                subview_shape));
    }

    auto subviews(const std::vector<std::size_t> &subview_shape) const
        requires dynamic_shape<Derived>
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (subview_shape.size() != this->rank()) {
                throw std::invalid_argument("Subview dimensions must match tensor rank");
            }
            if (std::accumulate(subview_shape.begin(), subview_shape.end(), 1ULL, std::multiplies<>()) !=
                this->size()) {
                throw std::invalid_argument("Subview dimensions must evenly divide tensor dimensions");
            }
        }

        return std::ranges::subrange(
            dynamic_subview_iterator<const Derived>(static_cast<const Derived *>(this),
                                                    std::vector<std::size_t>(this->rank(), 0), subview_shape),
            dynamic_subview_iterator<const Derived>(
                static_cast<const Derived *>(this),
                [this, &subview_shape] {
                    auto end_indices = this->shape();
                    std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                                   std::divides<>());
                    return end_indices;
                }(),
                subview_shape));
    }
};

} // namespace squint

#endif // SQUINT_ITERABLE_TENSOR_HPP