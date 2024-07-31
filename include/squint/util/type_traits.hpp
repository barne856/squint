#ifndef SQUINT_UTIL_TYPE_TRAITS_HPP
#define SQUINT_UTIL_TYPE_TRAITS_HPP

#include <array>
#include <type_traits>

namespace squint {

// Helper type trait for checking if a type is an index sequence
template <typename T> struct is_index_sequence : std::false_type {};
template <std::size_t... Dims> struct is_index_sequence<std::index_sequence<Dims...>> : std::true_type {};

} // namespace squint

#endif // SQUINT_UTIL_TYPE_TRAITS_HPP