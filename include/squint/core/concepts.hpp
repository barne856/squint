#ifndef SQUINT_CORE_CONCEPTS_HPP
#define SQUINT_CORE_CONCEPTS_HPP

#include "squint/core/layout.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/util/type_traits.hpp"

#include <array>
#include <concepts>
#include <type_traits>
#include <vector>

namespace squint {

// Base tensor concept
template <typename T>
concept tensorial = requires(T t) {
    typename T::value_type;
    typename T::shape_type;
    { t.rank() } -> std::convertible_to<std::size_t>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::convertible_to<std::vector<std::size_t>>;
    { t.data() } -> std::convertible_to<typename T::value_type *>;
    { T::layout() } -> std::same_as<layout>;
    { T::error_checking() } -> std::same_as<error_checking>;
};

// Scalar concepts
template <typename T>
concept floating_point = std::is_floating_point_v<T>;
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;
template <typename T>
concept quantitative = requires(T t) {
    typename T::value_type;
    typename T::dimension_type;
    requires arithmetic<typename T::value_type>;
};
template <typename T>
concept scalar_like = arithmetic<T> || quantitative<T>;

// Fixed shape concept
template <typename T>
concept fixed_shape = tensorial<T> && is_index_sequence<typename T::shape_type>::value;

// Dynamic shape concept
template <typename T>
concept dynamic_shape = tensorial<T> && std::is_same_v<typename T::shape_type, std::vector<std::size_t>> &&
                        requires(T t, const std::vector<std::size_t> &new_shape) {
                            { t.reshape(new_shape) } -> std::same_as<void>;
                        };

// Mutability concept
template <typename T>
concept mutable_tensor =
    tensorial<T> && requires(T t, typename T::value_type v, const std::vector<std::size_t> &indices) {
        { t.at(indices) } -> std::same_as<typename T::value_type &>;
        { t.at(indices) = v } -> std::same_as<typename T::value_type &>;
    };

template <typename T>
concept const_tensor = tensorial<T> && !mutable_tensor<T>;

// View concept
template <typename T>
concept tensor_view_like = tensorial<T> && requires(T t) {
    typename T::stride_type;
    { t.strides() } -> std::convertible_to<typename T::stride_type>;
    { t.shape() } -> std::convertible_to<std::vector<std::size_t>>;
};

// Element-wise addition concept
template <typename T>
concept addable = tensorial<T> && requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a += b } -> std::same_as<T &>;
};

// Element-wise subtraction concept
template <typename T>
concept subtractable = tensorial<T> && requires(T a, T b) {
    { a - b } -> std::same_as<T>;
    { a -= b } -> std::same_as<T &>;
};

// Scalar multiplication concept
template <typename T, typename U>
concept scalar_multiplicable = tensorial<T> && scalar_like<U> && requires(T a, U s) {
    { a *s } -> tensorial;
    { s *a } -> tensorial;
};

// Scalar division concept
template <typename T, typename U>
concept scalar_divisible = tensorial<T> && scalar_like<U> && requires(T a, U s) {
    { a / s } -> tensorial;
};

// Tensor multiplication concept
template <typename T, typename U>
concept tensor_multiplicable = tensorial<T> && tensorial<U> && requires(T a, U b) {
    { a *b } -> tensorial;
};

// Tensor division concept (for solving linear systems)
template <typename T, typename U>
concept tensor_divisible = tensorial<T> && tensorial<U> && requires(T a, U b) {
    { a / b } -> tensorial;
};

// Combined linear algebraic concept
template <typename T>
concept algebraic = tensorial<T> && addable<T> && subtractable<T> && scalar_multiplicable<T, typename T::value_type> &&
                    scalar_divisible<T, typename T::value_type> && tensor_multiplicable<T, T> && tensor_divisible<T, T>;

// Reshaping concept
template <typename T>
concept reshapable = dynamic_shape<T>;

// Subview concept
template <typename T>
concept subviewable =
    tensorial<T> && requires(T t, const std::vector<std::size_t> &shape, const std::vector<std::size_t> &start) {
        { t.subview(shape, start) } -> tensorial;
    };

} // namespace squint

#endif // SQUINT_CORE_CONCEPTS_HPP