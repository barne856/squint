#ifndef SQUINT_CORE_CONCEPTS_HPP
#define SQUINT_CORE_CONCEPTS_HPP

#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/util/type_traits.hpp"

#include <concepts>
#include <ratio>
#include <type_traits>
#include <vector>

namespace squint {

/**
 * @concept rational
 * @brief Concept for the standard library ratio template type.
 *
 * This concept ensures that a type T is an instance of std::ratio.
 *
 * @tparam T The type to check against the rational concept.
 */
template <class T>
concept rational = std::is_same<T, std::ratio<T::num, T::den>>::value;

/**
 * @concept dimensional
 * @brief Concept for a physical dimension.
 *
 * This concept defines the requirements for a type to be considered a valid
 * physical dimension. It must have rational exponents for each of the seven
 * SI base dimensions.
 *
 * @tparam U The type to check against the dimensional concept.
 */
template <class U>
concept dimensional = requires {
    requires rational<typename U::L>; ///< Length
    requires rational<typename U::T>; ///< Time
    requires rational<typename U::M>; ///< Mass
    requires rational<typename U::K>; ///< Temperature
    requires rational<typename U::I>; ///< Current
    requires rational<typename U::N>; ///< Amount of substance
    requires rational<typename U::J>; ///< Luminous intensity
};

/**
 * @concept dimensionless
 * @brief Concept for a dimensionless physical dimension.
 *
 * This concept checks if all dimension exponents are zero.
 *
 * @tparam U The type to check against the dimensionless concept.
 */
template <class U>
concept dimensionless = dimensional<U> && (U::L::num == 0) && (U::T::num == 0) && (U::M::num == 0) &&
                        (U::K::num == 0) && (U::I::num == 0) && (U::N::num == 0) && (U::J::num == 0);

/**
 * @concept tensorial
 * @brief Base concept for tensor-like types.
 *
 * This concept defines the basic requirements for a type to be considered a tensor.
 *
 * @tparam T The type to check against the tensorial concept.
 */
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

/**
 * @concept floating_point
 * @brief Concept for floating-point types.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept floating_point = std::is_floating_point_v<T>;

/**
 * @concept arithmetic
 * @brief Concept for arithmetic types.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

/**
 * @concept quantitative
 * @brief Concept for quantity types.
 *
 * This concept defines the requirements for a type to be considered a quantity.
 *
 * @tparam T The type to check against the quantitative concept.
 */
template <typename T>
concept quantitative = requires(T t) {
    typename T::value_type;
    typename T::dimension_type;
    requires arithmetic<typename T::value_type>;
};

/**
 * @concept scalar_like
 * @brief Concept for scalar-like types.
 *
 * This concept includes both arithmetic types and quantitative types.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept scalar_like = arithmetic<T> || quantitative<T>;

/**
 * @concept dimensionless_quantity
 * @brief Concept for dimensionless quantity types.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept dimensionless_quantity = quantitative<T> && dimensionless<typename T::dimension_type>;

/**
 * @concept dimensionless_scalar
 * @brief Concept for dimensionless scalar types.
 *
 * This concept includes both arithmetic types and dimensionless quantity types.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept dimensionless_scalar = arithmetic<T> || dimensionless_quantity<T>;

/**
 * @concept fixed_shape
 * @brief Concept for tensors with a fixed shape.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept fixed_shape = tensorial<T> && is_index_sequence<typename T::shape_type>::value;

/**
 * @concept dynamic_shape
 * @brief Concept for tensors with a dynamic shape.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept dynamic_shape = tensorial<T> && std::is_same_v<typename T::shape_type, std::vector<std::size_t>> &&
                        requires(T t, const std::vector<std::size_t> &new_shape) {
                            { t.reshape(new_shape) } -> std::same_as<void>;
                        };

/**
 * @concept mutable_tensor
 * @brief Concept for mutable tensors.
 *
 * This concept defines the requirements for a tensor type to be considered mutable.
 *
 * @tparam T The type to check against the mutable_tensor concept.
 */
template <typename T>
concept mutable_tensor =
    tensorial<T> && requires(T t, typename T::value_type v, const std::vector<std::size_t> &indices) {
        { t.at(indices) } -> std::same_as<typename T::value_type &>;
        { t.at(indices) = v } -> std::same_as<typename T::value_type &>;
    };

/**
 * @concept const_tensor
 * @brief Concept for constant tensors.
 *
 * This concept defines tensors that are not mutable.
 *
 * @tparam T The type to check against the const_tensor concept.
 */
template <typename T>
concept const_tensor = tensorial<T> && !mutable_tensor<T>;

/**
 * @concept tensor_view_like
 * @brief Concept for tensor view types.
 *
 * This concept defines the requirements for a type to be considered a tensor view.
 *
 * @tparam T The type to check against the tensor_view_like concept.
 */
template <typename T>
concept tensor_view_like = tensorial<T> && requires(T t) {
    typename T::stride_type;
    { t.strides() } -> std::convertible_to<typename T::stride_type>;
    { t.shape() } -> std::convertible_to<std::vector<std::size_t>>;
};

/**
 * @concept addable
 * @brief Concept for tensors that support element-wise addition.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept addable = tensorial<T> && requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a += b } -> std::same_as<T &>;
};

/**
 * @concept subtractable
 * @brief Concept for tensors that support element-wise subtraction.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept subtractable = tensorial<T> && requires(T a, T b) {
    { a - b } -> std::same_as<T>;
    { a -= b } -> std::same_as<T &>;
};

/**
 * @concept scalar_multiplicable
 * @brief Concept for tensors that support scalar multiplication.
 *
 * @tparam T The tensor type.
 * @tparam U The scalar type.
 */
template <typename T, typename U>
concept scalar_multiplicable = tensorial<T> && scalar_like<U> && requires(T a, U s) {
    { a *s } -> tensorial;
    { s *a } -> tensorial;
};

/**
 * @concept scalar_divisible
 * @brief Concept for tensors that support scalar division.
 *
 * @tparam T The tensor type.
 * @tparam U The scalar type.
 */
template <typename T, typename U>
concept scalar_divisible = tensorial<T> && scalar_like<U> && requires(T a, U s) {
    { a / s } -> tensorial;
};

/**
 * @concept tensor_multiplicable
 * @brief Concept for tensors that support tensor multiplication.
 *
 * @tparam T The first tensor type.
 * @tparam U The second tensor type.
 */
template <typename T, typename U>
concept tensor_multiplicable = tensorial<T> && tensorial<U> && requires(T a, U b) {
    { a *b } -> tensorial;
};

/**
 * @concept tensor_divisible
 * @brief Concept for tensors that support tensor division (for solving linear systems).
 *
 * @tparam T The dividend tensor type.
 * @tparam U The divisor tensor type.
 */
template <typename T, typename U>
concept tensor_divisible = tensorial<T> && tensorial<U> && requires(T a, U b) {
    { a / b } -> tensorial;
};

/**
 * @concept algebraic
 * @brief Concept for tensors that support all basic algebraic operations.
 *
 * This concept combines requirements for addition, subtraction, scalar multiplication,
 * scalar division, tensor multiplication, and tensor division.
 *
 * @tparam T The type to check against the algebraic concept.
 */
template <typename T>
concept algebraic = tensorial<T> && addable<T> && subtractable<T> && scalar_multiplicable<T, typename T::value_type> &&
                    scalar_divisible<T, typename T::value_type> && tensor_multiplicable<T, T> && tensor_divisible<T, T>;

/**
 * @concept reshapable
 * @brief Concept for tensors that can be reshaped.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept reshapable = dynamic_shape<T>;

/**
 * @concept subviewable
 * @brief Concept for tensors that support creating subviews.
 *
 * @tparam T The type to check against the subviewable concept.
 */
template <typename T>
concept subviewable =
    tensorial<T> && requires(T t, const std::vector<std::size_t> &shape, const std::vector<std::size_t> &start) {
        { t.subview(shape, start) } -> tensorial;
    };

} // namespace squint

#endif // SQUINT_CORE_CONCEPTS_HPP