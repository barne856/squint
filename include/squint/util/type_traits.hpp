/**
 * @file type_traits.hpp
 * @brief Type traits utilities for the Squint library.
 *
 * This file provides type traits utilities, including a trait for
 * checking if a type is an index sequence.
 */

#ifndef SQUINT_UTIL_TYPE_TRAITS_HPP
#define SQUINT_UTIL_TYPE_TRAITS_HPP

#include <type_traits>
#include <utility>

namespace squint {

/**
 * @brief Type trait to check if a type is an index sequence.
 *
 * This trait can be used to determine at compile-time whether a given type
 * is an instantiation of std::index_sequence.
 *
 * @tparam T The type to check.
 */
template <typename T> struct is_index_sequence : std::false_type {};

/**
 * @brief Specialization of is_index_sequence for actual index sequences.
 *
 * This specialization inherits from std::true_type for types that are
 * instantiations of std::index_sequence.
 *
 * @tparam Dims Variadic template parameter for the indices in the sequence.
 */
template <std::size_t... Dims> struct is_index_sequence<std::index_sequence<Dims...>> : std::true_type {};

} // namespace squint

#endif // SQUINT_UTIL_TYPE_TRAITS_HPP