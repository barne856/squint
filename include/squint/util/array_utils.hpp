/**
 * @file array_utils.hpp
 * @brief Utility functions for working with arrays and index sequences.
 *
 * This file provides utility functions for creating arrays from index sequences
 * and computing products of elements in index sequences.
 */

#ifndef SQUINT_UTIL_ARRAY_UTILS_HPP
#define SQUINT_UTIL_ARRAY_UTILS_HPP

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
template <std::size_t... Ix> constexpr std::size_t product(std::index_sequence<Ix...> /*unused*/) { return (Ix * ...); }

} // namespace squint

#endif // SQUINT_UTIL_ARRAY_UTILS_HPP