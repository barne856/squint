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
template <std::size_t Begin, std::size_t End, std::size_t... Dims> struct product_range {
    static constexpr std::size_t value = 1;
};

// Base case for product_range
template <std::size_t Begin, std::size_t End, std::size_t First, std::size_t... Rest>
struct product_range<Begin, End, First, Rest...> {
    static constexpr std::size_t value = (Begin < End ? First : 1) * product_range<Begin + 1, End, Rest...>::value;
};

/**
 * @brief Helper to compute the stride of a single dimension.
 *
 * This template computes the stride of a single dimension at index Idx
 * with optional dimensions Dims. The stride is computed as the product of
 * all dimensions before the index Idx in Dims.
 *
 */

template <std::size_t Idx, layout Layout, std::size_t... Dims> struct compute_single_stride {
    static constexpr std::size_t value = Layout == layout::row_major
                                             ? product_range<Idx + 1, sizeof...(Dims), Dims...>::value
                                             : product_range<0, Idx, Dims...>::value;
};

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
    using type = std::index_sequence<compute_single_stride<Is, Layout, Dims...>::value...>;
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

} // namespace squint

#endif // SQUINT_CORE_LAYOUT_HPP