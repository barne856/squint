#ifndef SQUINT_TENSOR_TENSOR_TYPES_HPP
#define SQUINT_TENSOR_TENSOR_TYPES_HPP

#include "squint/tensor/tensor.hpp"

#include <cstddef>
#include <utility>

namespace squint {

/**
 * @brief Type alias for creating tensors with specific type and dimensions.
 * @tparam T The data type of the tensor elements.
 * @tparam Dims The dimensions of the tensor.
 */
template <typename T, std::size_t... Dims> using tens_t = tensor<T, std::index_sequence<Dims...>>;

/**
 * @brief Type alias for creating float tensors with specific dimensions.
 * @tparam Dims The dimensions of the tensor.
 */
template <std::size_t... Dims> using tens = tens_t<float, Dims...>;

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_TYPES_HPP