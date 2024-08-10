/**
 * @file array_utils.hpp
 * @brief Utility functions for working with arrays and index sequences.
 *
 * This file provides utility functions for creating arrays from index sequences
 * and computing products of elements in index sequences.
 */

#ifndef SQUINT_UTIL_ARRAY_UTILS_HPP
#define SQUINT_UTIL_ARRAY_UTILS_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/layout.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace squint {

/**
 * @brief Creates an array from an index sequence.
 *
 * This function takes an index sequence and returns an std::array
 * containing the indices as elements.
 *
 * @tparam Ix Variadic template parameter for indices.
 * @param unused An index sequence (unused, only used for type deduction).
 * @return std::array<std::size_t, sizeof...(Ix)> An array containing the indices.
 */
template <std::size_t... Ix> constexpr auto make_array(std::index_sequence<Ix...> /*unused*/) {
    return std::array{Ix...};
}

/**
 * @brief Computes the product of elements in an index sequence.
 *
 * This function multiplies all the indices in the given index sequence.
 *
 * @tparam Ix Variadic template parameter for indices.
 * @param unused An index sequence (unused, only used for type deduction).
 * @return std::size_t The product of all indices in the sequence.
 */
template <std::size_t... Ix> constexpr auto product(std::index_sequence<Ix...> /*unused*/) -> std::size_t {
    return (Ix * ...);
}

// Helper function to check if tensor dimensions are divisible by subview dimensions
template <fixed_tensor T, typename SubviewShape> constexpr auto dimensions_divisible() -> bool {
    constexpr auto shape_arr = make_array(typename T::shape_type{});
    constexpr auto subview_arr = make_array(SubviewShape{});
    for (std::size_t i = 0; i < shape_arr.size(); ++i) {
        if (shape_arr[i] % subview_arr[i] != 0) {
            return false;
        }
    }
    return true;
}

// Helper to remove the first element from a sequence
template <typename Sequence> struct tail_sequence;

template <std::size_t First, std::size_t... Rest> struct tail_sequence<std::index_sequence<First, Rest...>> {
    using type = std::index_sequence<Rest...>;
};

// Alias template for tail_sequence
template <typename Sequence> using tail_sequence_t = typename tail_sequence<Sequence>::type;

// Helper to add an element to the beginning of a sequence
template <typename Sequence, std::size_t New> struct prepend_sequence;

template <std::size_t... Rest, std::size_t New> struct prepend_sequence<std::index_sequence<Rest...>, New> {
    using type = std::index_sequence<New, Rest...>;
};

// Alias template for prepend_sequence
template <typename Sequence, std::size_t New> using prepend_sequence_t = typename prepend_sequence<Sequence, New>::type;
} // namespace squint

#endif // SQUINT_UTIL_ARRAY_UTILS_HPP