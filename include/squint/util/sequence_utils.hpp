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
#include <type_traits>
#include <utility>
#include <algorithm>

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
 * @brief Checks if all elements of an index sequence are less than a given value.
 * @tparam Ix Variadic template parameter for indices.
 * @param max_value The maximum value to compare against.
 * @return bool True if all elements are less than max_value, false otherwise.
 */
template <std::size_t... Ix> constexpr auto all_less_than(std::index_sequence<Ix...> /*unused*/, std::size_t max_value)
    -> bool {
    return ((Ix < max_value) && ...);
}

// max of sequence
template <std::size_t... Ix> constexpr auto max(std::index_sequence<Ix...> /*unused*/) -> std::size_t {
    return std::max({Ix...});
}

// Helper to concatenate two index sequences
template <typename Sequence1, typename Sequence2> struct concat_sequence;

template <std::size_t... Ns1, std::size_t... Ns2>
struct concat_sequence<std::index_sequence<Ns1...>, std::index_sequence<Ns2...>> {
    using type = std::index_sequence<Ns1..., Ns2...>;
};

// Alias template for concat_sequence
template <typename Sequence1, typename Sequence2> using concat_sequence_t = typename concat_sequence<Sequence1, Sequence2>::type;

// Helper to remove the first element from a sequence
template <typename Sequence> struct tail_sequence;

template <std::size_t First, std::size_t... Rest> struct tail_sequence<std::index_sequence<First, Rest...>> {
    using type = std::index_sequence<Rest...>;
};

// Alias template for tail_sequence
template <typename Sequence> using tail_sequence_t = typename tail_sequence<Sequence>::type;

// Helper to remove the last element from a sequence
template <typename Sequence> struct init_sequence {
    template <typename S, std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>)
        -> std::index_sequence<std::get<Ns>(make_array(S{}))...>;

    using type = decltype(helper<Sequence>(std::make_index_sequence<Sequence::size() - 1>{}));
};

// Alias template for init_sequence
template <typename Sequence> using init_sequence_t = typename init_sequence<Sequence>::type;

// Helper to add an element to the beginning of a sequence
template <typename Sequence, std::size_t New> struct prepend_sequence;

template <std::size_t... Rest, std::size_t New> struct prepend_sequence<std::index_sequence<Rest...>, New> {
    using type = std::index_sequence<New, Rest...>;
};

// Alias template for prepend_sequence
template <typename Sequence, std::size_t New> using prepend_sequence_t = typename prepend_sequence<Sequence, New>::type;

// Helper to add an element to the beginning of a sequence
template <typename Sequence, std::size_t New> struct append_sequence;

// Helper to append an element to a sequence
template <std::size_t... Rest, std::size_t New> struct append_sequence<std::index_sequence<Rest...>, New>{
    using type = std::index_sequence<Rest..., New>;
};

// Alias template for append_sequence
template <typename Sequence, std::size_t New> using append_sequence_t = typename append_sequence<Sequence, New>::type;

// Helper to remove the last N elements from a sequence
template <typename Sequence, std::size_t N> struct remove_last_n {
    static_assert(N <= Sequence::size(), "Cannot remove more elements than the sequence contains");

    template <typename S, std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>) -> std::index_sequence<std::get<Ns>(make_array(S{}))...>;

    using type = decltype(helper<Sequence>(std::make_index_sequence<Sequence::size() - N>{}));
};

// Alias template for remove_last_n
template <typename Sequence, std::size_t N> using remove_last_n_t = typename remove_last_n<Sequence, N>::type;

// Helper to reverse an index sequence
template <typename Sequence> struct reverse_sequence{
    template <typename S, std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>)
        -> std::index_sequence<std::get<Sequence::size() - 1 - Ns>(make_array(S{}))...>;

    using type = decltype(helper<Sequence>(std::make_index_sequence<Sequence::size()>{}));
};

// Alias template for reverse_sequence
template <typename Sequence> using reverse_sequence_t = typename reverse_sequence<Sequence>::type;

// Helper to make repeating index sequences of a single value and length N
template <std::size_t N, std::size_t Value> struct repeat_sequence {
    template <std::size_t... Ns> static auto helper(std::index_sequence<Ns...>) -> std::index_sequence<(Value + Ns*0)...>;

    using type = decltype(helper(std::make_index_sequence<N>{}));
};

// Alias template for repeat_sequence
template <std::size_t N, std::size_t Value> using repeat_sequence_t = typename repeat_sequence<N, Value>::type;

// checks if a sequence has no duplicates
template <typename Sequence> constexpr bool is_unique() {
    auto arr = make_array(Sequence{});
    return static_cast<bool>(std::unique(arr.begin(), arr.end()) == arr.end());
}

// Concept for valid index permutation used to transpose a tensor.
// All indices must have no duplicates, and be less than the total number of indices.
template <typename Sequence>
concept valid_index_permutation =
    is_unique<Sequence>() && all_less_than(Sequence{}, Sequence::size());


// Apply index permutation to a sequence
template <typename Sequence, typename IndexPermutation, std::size_t pad_value>
struct apply_permutation {
    template <typename S, std::size_t... Ns>
    static auto helper(std::index_sequence<Ns...>)
        -> std::index_sequence<std::get<Ns>(make_array(S{}))...>;

    using type = decltype(helper<concat_sequence_t<Sequence,repeat_sequence_t<IndexPermutation::size() - Sequence::size(), pad_value>>>(IndexPermutation{}));
};

template <typename Sequence, typename IndexPermutation, std::size_t pad_value>
using apply_permutation_t = typename apply_permutation<Sequence, IndexPermutation, pad_value>::type;

// Helper to determine if two sequences representing tensor shapes can be implicitly converted
template <typename Sequence1, typename Sequence2>
struct implicit_convertible_shapes {
    static constexpr bool helper() {
        constexpr auto arr1 = make_array(Sequence1{});
        constexpr auto arr2 = make_array(Sequence2{});
        constexpr std::size_t size1 = arr1.size();
        constexpr std::size_t size2 = arr2.size();
        constexpr std::size_t min_size = std::min(size1, size2);

        // Check if the common elements are the same
        for (std::size_t i = 0; i < min_size; ++i) {
            if (arr1[i] != arr2[i]) return false;
        }

        // Check if the extra elements in the longer sequence are all 1's
        if (size1 > size2) {
            for (std::size_t i = size2; i < size1; ++i) {
                if (arr1[i] != 1) return false;
            }
        } else if (size2 > size1) {
            for (std::size_t i = size1; i < size2; ++i) {
                if (arr2[i] != 1) return false;
            }
        }

        return true;
    }

    static constexpr bool value = helper();
};

// Alias template for implicit_convertible_shapes
template <typename Sequence1, typename Sequence2>
inline constexpr bool implicit_convertible_shapes_v = implicit_convertible_shapes<Sequence1, Sequence2>::value;

// Helper to determine if two sequences representing tensor strides can be implicitly converted
template <typename Sequence1, typename Sequence2>
struct implicit_convertible_strides {
    static constexpr bool helper() {
        constexpr auto arr1 = make_array(Sequence1{});
        constexpr auto arr2 = make_array(Sequence2{});
        constexpr std::size_t size1 = arr1.size();
        constexpr std::size_t size2 = arr2.size();
        constexpr std::size_t min_size = std::min(size1, size2);

        // Check if the common elements are the same
        for (std::size_t i = 0; i < min_size; ++i) {
            if (arr1[i] != arr2[i]) return false;
        }

        return true;
    }

    static constexpr bool value = helper();
};

// Alias template for implicit_convertible_strides
template <typename Sequence1, typename Sequence2>
inline constexpr bool implicit_convertible_strides_v = implicit_convertible_strides<Sequence1, Sequence2>::value;

} // namespace squint

#endif // SQUINT_UTIL_SEQUENCE_UTILS_HPP