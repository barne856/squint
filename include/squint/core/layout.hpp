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

} // namespace squint

#endif // SQUINT_CORE_LAYOUT_HPP