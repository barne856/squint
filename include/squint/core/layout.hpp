/**
 * @file layout.hpp
 * @brief Defines memory layout options for tensors.
 *
 * This file provides an enumeration for specifying the memory layout
 * of tensors in the Squint library. The layout determines how multi-dimensional
 * tensor data is arranged in contiguous memory.
 */

#ifndef SQUINT_CORE_LAYOUT_HPP
#define SQUINT_CORE_LAYOUT_HPP

#include "squint/util/sequence_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace squint {

/**
 * @brief Enumeration to specify the memory layout of tensors.
 *
 * This enum class is used as a template parameter to control how tensor data
 * is arranged in memory. It affects the performance of certain operations
 * and the compatibility with external libraries.
 */
enum class layout : uint8_t {
    row_major,   /**< Row-major layout: elements of a row are contiguous in memory */
    column_major /**< Column-major layout: elements of a column are contiguous in memory */
};

/**
 * @brief Helper to compute the product of a range of integers.
 *
 * This template computes the product of a range of integers [Begin, End)
 * with optional dimensions Dims. The product is computed as the product of
 * all dimensions in the range [Begin, End) and the dimensions in Dims.
 *
 */
template <std::size_t Start, std::size_t End, typename Sequence> struct product_range;

template <std::size_t Start, std::size_t End, std::size_t... Dims>
struct product_range<Start, End, std::index_sequence<Dims...>> {
    template <std::size_t... Is> static constexpr auto compute(std::index_sequence<Is...> /*unused*/) -> std::size_t {
        return product(std::index_sequence<std::get<Start + Is>(make_array(std::index_sequence<Dims...>{}))...>{});
    }

    static constexpr std::size_t value = compute(std::make_index_sequence<End - Start>{});
};

/**
 * @brief Helper to compute the stride of a single dimension.
 *
 * This template computes the stride of a single dimension at index Idx
 * with optional dimensions Dims. The stride is computed as the product of
 * all dimensions before the index Idx in Dims.
 *
 */
template <std::size_t Idx, layout Layout, typename Shape> struct compute_single_stride;
// NOLINTBEGIN
template <std::size_t Idx, layout Layout, std::size_t... Dims>
struct compute_single_stride<Idx, Layout, std::index_sequence<Dims...>> {
    using reversed_shape = reverse_sequence_t<std::index_sequence<Dims...>>;

    static constexpr std::size_t value =
        Layout == layout::row_major
            ? (Idx == sizeof...(Dims) - 1 ? 1 : product_range<0, sizeof...(Dims) - Idx - 1, reversed_shape>::value)
            : product_range<0, Idx, std::index_sequence<Dims...>>::value;
};
// NOLINTEND

/**
 * @brief Helper to compute the strides of a tensor shape.
 *
 * This template computes the strides of a tensor shape with optional layout.
 * The strides are computed as the product of all dimensions before the current
 * dimension in the specified layout.
 *
 */
template <layout Layout, typename Seq, typename Shape> struct compute_strides;

template <layout Layout, std::size_t... Is, std::size_t... Dims>
struct compute_strides<Layout, std::index_sequence<Is...>, std::index_sequence<Dims...>> {
    using type = std::index_sequence<compute_single_stride<Is, Layout, std::index_sequence<Dims...>>::value...>;
};

/**
 * @brief Helper to compute the strides of a tensor shape.
 *
 * This template computes the strides of a tensor shape with optional layout.
 * The strides are computed as the product of all dimensions before the current
 * dimension in the specified layout.
 *
 */
struct strides {
    template <typename Shape>
    using row_major = typename compute_strides<layout::row_major, std::make_index_sequence<Shape::size()>, Shape>::type;
    template <typename Shape>
    using column_major =
        typename compute_strides<layout::column_major, std::make_index_sequence<Shape::size()>, Shape>::type;
};

template <std::size_t... Dims> using seq = std::index_sequence<Dims...>; //< Compile-time sequence
template <std::size_t... Dims> using shape = seq<Dims...>;               //< Compile-time shape
using dynamic = std::vector<std::size_t>;                                //< Dynamic shape

// helper struct for is column major strides
template <typename Strides, typename Shape> struct is_column_major_t;

template <std::size_t... Strides, std::size_t... Shape>
struct is_column_major_t<std::index_sequence<Strides...>, std::index_sequence<Shape...>> {
    static constexpr bool value =
        std::is_same_v<std::index_sequence<Strides...>, strides::column_major<shape<Shape...>>>;
};

template <typename Strides, typename Shape>
inline constexpr bool is_column_major_v = is_column_major_t<Strides, Shape>::value;

// helper struct for is row major strides
template <typename Strides, typename Shape> struct is_row_major_t;

template <std::size_t... Strides, std::size_t... Shape>
struct is_row_major_t<std::index_sequence<Strides...>, std::index_sequence<Shape...>> {
    static constexpr bool value = std::is_same_v<std::index_sequence<Strides...>, strides::row_major<shape<Shape...>>>;
};

template <typename Strides, typename Shape>
inline constexpr bool is_row_major_v = is_row_major_t<Strides, Shape>::value;

} // namespace squint

#endif // SQUINT_CORE_LAYOUT_HPP