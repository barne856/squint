/**
 * @file sequence_utils.hpp
 * @brief Utility functions and type traits for compile-time sequence operations.
 *
 * This file provides a set of template metaprogramming utilities for working with
 * std::index_sequence and similar compile-time sequences. It includes type traits,
 * sequence manipulations, and arithmetic operations on sequences.
 *
 */

#ifndef SQUINT_UTIL_SEQUENCE_UTILS_HPP
#define SQUINT_UTIL_SEQUENCE_UTILS_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace squint {

/**
 * @brief Type trait to check if a type is an index sequence.
 * @tparam T The type to check.
 */
template <typename T> struct is_index_sequence : std::false_type {};

template <std::size_t... Dims> struct is_index_sequence<std::index_sequence<Dims...>> : std::true_type {};

/**
 * @brief Creates an array from an index sequence.
 * @tparam Ix Variadic template parameter for indices.
 * @return std::array<std::size_t, sizeof...(Ix)> An array containing the indices.
 */
template <std::size_t... Ix> constexpr auto make_array(std::index_sequence<Ix...> /*unused*/) {
    return std::array{Ix...};
}

/**
 * @brief Computes the product of elements in an index sequence.
 * @tparam Ix Variadic template parameter for indices.
 * @return std::size_t The product of all indices in the sequence.
 */
template <std::size_t... Ix> constexpr auto product(std::index_sequence<Ix...> /*seq*/) { return (Ix * ...); }

/**
 * @brief Computes the sum of elements in an index sequence.
 * @tparam Ix Variadic template parameter for indices.
 * @return std::size_t The sum of all indices in the sequence.
 */
template <std::size_t... Ix> constexpr auto sum(std::index_sequence<Ix...> /*seq*/) { return (Ix + ...); }

/**
 * @brief Checks if all elements of an index sequence are equal.
 * @tparam Ix Variadic template parameter for indices.
 * @return bool True if all elements are equal, false otherwise.
 */
template <std::size_t... Ix> constexpr auto all_equal(std::index_sequence<Ix...> /*unused*/) -> bool {
    if constexpr (sizeof...(Ix) == 0) {
        // An empty sequence is considered to have all elements equal
        return true;
    } else {
        // Use fold expression to compare all elements with the first one
        return ((Ix == std::get<0>(std::tuple{Ix...})) && ...);
    }
}

/**
 * @brief Base template for sequence manipulation operations.
 * @tparam Op The specific operation to perform.
 * @tparam Sequence The input sequence type.
 */
template <typename Op, typename Sequence> struct sequence_op;

/**
 * @brief Removes the first element from a sequence.
 */
struct tail_op {};

/**
 * @brief Removes the last element from a sequence.
 */
struct init_op {};

/**
 * @brief Reverses a sequence.
 */
struct reverse_op {};

// Specializations for each operation
template <std::size_t First, std::size_t... Rest> struct sequence_op<tail_op, std::index_sequence<First, Rest...>> {
    using type = std::index_sequence<Rest...>;
};

template <std::size_t... Is> struct sequence_op<init_op, std::index_sequence<Is...>> {
    template <std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>)
        -> std::index_sequence<std::get<Ns>(make_array(std::index_sequence<Is...>{}))...>;

    using type = decltype(helper(std::make_index_sequence<sizeof...(Is) - 1>{}));
};

template <std::size_t... Is> struct sequence_op<reverse_op, std::index_sequence<Is...>> {
    template <std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>)
        -> std::index_sequence<std::get<sizeof...(Is) - 1 - Ns>(std::array{Is...})...>;

    using type = decltype(helper(std::make_index_sequence<sizeof...(Is)>{}));
};

/**
 * @brief Alias template for sequence operations.
 * @tparam Op The operation to perform.
 * @tparam Sequence The input sequence type.
 */
template <typename Op, typename Sequence> using sequence_op_t = typename sequence_op<Op, Sequence>::type;

/**
 * @brief Alias for tail operation.
 * @tparam Sequence The input sequence type.
 */
template <typename Sequence> using tail_sequence_t = sequence_op_t<tail_op, Sequence>;

/**
 * @brief Alias for init operation.
 * @tparam Sequence The input sequence type.
 */
template <typename Sequence> using init_sequence_t = sequence_op_t<init_op, Sequence>;

/**
 * @brief Alias for reverse operation.
 * @tparam Sequence The input sequence type.
 */
template <typename Sequence> using reverse_sequence_t = sequence_op_t<reverse_op, Sequence>;

/**
 * @brief Prepends an element to a sequence.
 * @tparam Sequence The input sequence type.
 * @tparam New The element to prepend.
 */
template <typename Sequence, std::size_t New> struct prepend_sequence {
    using type = decltype(std::index_sequence<New>() + std::declval<Sequence>());
};

/**
 * @brief Alias for prepend operation.
 * @tparam Sequence The input sequence type.
 * @tparam New The element to prepend.
 */
template <typename Sequence, std::size_t New> using prepend_sequence_t = typename prepend_sequence<Sequence, New>::type;

/**
 * @brief Appends an element to a sequence.
 * @tparam Sequence The input sequence type.
 * @tparam New The element to append.
 */
template <typename Sequence, std::size_t New> struct append_sequence {
    using type = decltype(std::declval<Sequence>() + std::index_sequence<New>());
};

/**
 * @brief Alias for append operation.
 * @tparam Sequence The input sequence type.
 * @tparam New The element to append.
 */
template <typename Sequence, std::size_t New> using append_sequence_t = typename append_sequence<Sequence, New>::type;

/**
 * @brief Removes the last N elements from a sequence.
 * @tparam Sequence The input sequence type.
 * @tparam N The number of elements to remove.
 */
template <typename Sequence, std::size_t N> struct remove_last_n {
    static_assert(N <= Sequence::size(), "Cannot remove more elements than the sequence contains");

    template <std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>) -> std::index_sequence<std::get<Ns>(make_array(Sequence{}))...>;

    using type = decltype(helper(std::make_index_sequence<Sequence::size() - N>{}));
};

/**
 * @brief Alias for remove_last_n operation.
 * @tparam Sequence The input sequence type.
 * @tparam N The number of elements to remove.
 */
template <typename Sequence, std::size_t N> using remove_last_n_t = typename remove_last_n<Sequence, N>::type;

} // namespace squint

#endif // SQUINT_UTIL_SEQUENCE_UTILS_HPP