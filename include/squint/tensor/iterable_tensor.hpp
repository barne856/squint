#ifndef SQUINT_ITERABLE_TENSOR_HPP
#define SQUINT_ITERABLE_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/tensor/flat_iterator.hpp"
#include "squint/tensor/subview_iterator.hpp"
#include "squint/util/array_utils.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <ranges>
#include <vector>

namespace squint {

// Mixin class for iterable tensors
template <typename Derived> class iterable_tensor {
  public:
    using iterator = flat_iterator<Derived>;
    using const_iterator = flat_iterator<const Derived>;

    auto begin() {
        typename Derived::index_type start_indices{};
        if constexpr (dynamic_shape<Derived>) {
            start_indices.resize(static_cast<Derived *>(this)->rank(), 0);
        }
        return iterator(static_cast<Derived *>(this), start_indices);
    }
    auto end() { return iterator(static_cast<Derived *>(this), static_cast<Derived *>(this)->shape()); }
    auto begin() const {
        typename Derived::index_type start_indices{};
        if constexpr (dynamic_shape<Derived>) {
            start_indices.resize(static_cast<const Derived *>(this)->rank(), 0);
        }
        return const_iterator(static_cast<const Derived *>(this), start_indices);
    }
    auto end() const {
        return const_iterator(static_cast<const Derived *>(this), static_cast<Derived *>(this)->shape());
    }
    auto cbegin() const { return begin(); }
    auto cend() const { return end(); }

    // Subview iteration for fixed shape tensors
    template <typename SubviewShape>
    auto subviews()
        requires fixed_shape<Derived>
    {
        static_assert(SubviewShape::size() == Derived::rank(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewShape>(),
                      "Subview dimensions must evenly divide tensor dimensions");

        auto derived = static_cast<Derived *>(this);

        return std::ranges::subrange(
            subview_iterator<Derived, SubviewShape>(derived, std::array<std::size_t, SubviewShape::size()>{}),
            subview_iterator<Derived, SubviewShape>(derived, [] {
                auto end_indices = derived->shape();
                constexpr auto subview_shape = make_array(SubviewShape{});
                std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                               std::divides<>());
                return end_indices;
            }()));
    }

    template <typename SubviewShape>
    auto subviews() const
        requires fixed_shape<Derived>
    {
        static_assert(SubviewShape::size() == Derived::rank(), "Subview dimensions must match tensor rank");
        static_assert(dimensions_divisible<Derived, SubviewShape>(),
                      "Subview dimensions must evenly divide tensor dimensions");

        auto derived = static_cast<const Derived *>(this);

        return std::ranges::subrange(
            subview_iterator<const Derived, SubviewShape>(derived, std::array<std::size_t, SubviewShape::size()>{}),
            subview_iterator<const Derived, SubviewShape>(derived, [] {
                auto end_indices = derived->shape();
                constexpr auto subview_shape = make_array(SubviewShape{});
                std::transform(end_indices.begin(), end_indices.end(), subview_shape.begin(), end_indices.begin(),
                               std::divides<>());
                return end_indices;
            }()));
    }

    // Subview iteration for dynamic shape tensors
    auto subviews(const std::vector<std::size_t> &subview_shape)
        requires dynamic_shape<Derived>
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

        return std::ranges::subrange(subview_iterator<Derived, std::vector<std::size_t>>(
                                         derived, std::vector<std::size_t>(derived->rank(), 0), subview_shape),
                                     subview_iterator<Derived, std::vector<std::size_t>>(
                                         derived,
                                         [this, &subview_shape] {
                                             auto end_indices = derived->shape();
                                             std::transform(end_indices.begin(), end_indices.end(),
                                                            subview_shape.begin(), end_indices.begin(),
                                                            std::divides<>());
                                             return end_indices;
                                         }(),
                                         subview_shape));
    }

    auto subviews(const std::vector<std::size_t> &subview_shape) const
        requires dynamic_shape<Derived>
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

        return std::ranges::subrange(subview_iterator<const Derived, std::vector<std::size_t>>(
                                         derived, std::vector<std::size_t>(derived->rank(), 0), subview_shape),
                                     subview_iterator<const Derived, std::vector<std::size_t>>(
                                         derived,
                                         [this, &subview_shape] {
                                             auto end_indices = derived->shape();
                                             std::transform(end_indices.begin(), end_indices.end(),
                                                            subview_shape.begin(), end_indices.begin(),
                                                            std::divides<>());
                                             return end_indices;
                                         }(),
                                         subview_shape));
    }
};

} // namespace squint

#endif // SQUINT_ITERABLE_TENSOR_HPP