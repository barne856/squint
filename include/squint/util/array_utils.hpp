#ifndef SQUINT_UTIL_ARRAY_UTILS_HPP
#define SQUINT_UTIL_ARRAY_UTILS_HPP

#include <array>
#include <cstddef>
#include <utility>

namespace squint {

template <std::size_t... Ix> constexpr auto make_array(std::index_sequence<Ix...> /*unused*/) {
    return std::array{Ix...};
}

// Helper function to compute the product of elements in an index sequence
template <std::size_t... Ix> constexpr std::size_t product(std::index_sequence<Ix...> /*unused*/) { return (Ix * ...); }

} // namespace squint

#endif // SQUINT_UTIL_ARRAY_UTILS_HPP