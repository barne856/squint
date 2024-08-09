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

#include <cstddef>
#include <utility>

namespace squint {

/**
 * @brief Enumeration to specify the memory layout of tensors.
 *
 * This enum class is used as a template parameter to control how tensor data
 * is arranged in memory. It affects the performance of certain operations
 * and the compatibility with external libraries.
 */
enum class layout {
    row_major,    /**< Row-major layout: elements of a row are contiguous in memory */
    column_major  /**< Column-major layout: elements of a column are contiguous in memory */
};

// Helper to compute the product of a range of values in a parameter pack
template <std::size_t Begin, std::size_t End, std::size_t... Dims> struct product_range {
    static constexpr std::size_t value = 1;
};

template <std::size_t Begin, std::size_t End, std::size_t First, std::size_t... Rest>
struct product_range<Begin, End, First, Rest...> {
    static constexpr std::size_t value = (Begin < End ? First : 1) * product_range<Begin + 1, End, Rest...>::value;
};

// Helper to compute a single stride
template <std::size_t Idx, layout Layout, std::size_t... Dims> struct compute_single_stride {
    static constexpr std::size_t value = Layout == layout::row_major
                                             ? product_range<Idx + 1, sizeof...(Dims), Dims...>::value
                                             : product_range<0, Idx, Dims...>::value;
};

// Helper to compute all strides
template <layout Layout, typename Seq, typename Shape> struct compute_strides;

template <layout Layout, std::size_t... Is, std::size_t... Dims>
struct compute_strides<Layout, std::index_sequence<Is...>, std::index_sequence<Dims...>> {
    using type = std::index_sequence<compute_single_stride<Is, Layout, Dims...>::value...>;
};

// alias for row-major strides
template <typename Shape> using row_major_strides = typename compute_strides<layout::row_major, std::make_index_sequence<Shape::size()>, Shape>::type;

// alias for column-major strides
template <typename Shape> using column_major_strides = typename compute_strides<layout::column_major, std::make_index_sequence<Shape::size()>, Shape>::type;


} // namespace squint

#endif // SQUINT_CORE_LAYOUT_HPP