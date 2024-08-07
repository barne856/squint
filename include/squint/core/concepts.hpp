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
    typename T::index_type;
    typename T::shape_type;
    typename T::strides_type;
    { t.rank() } -> std::convertible_to<std::size_t>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::convertible_to<typename T::index_type>;
    { t.strides() } -> std::convertible_to<typename T::index_type>;
    { t.data() } -> std::convertible_to<typename T::value_type *>;
    { t.data() } -> std::convertible_to<const typename T::value_type *>;
    { T::layout() } -> std::same_as<layout>;
    { T::error_checking() } -> std::same_as<error_checking>;
    { t[std::declval<typename T::index_type>()] } -> std::convertible_to<typename T::value_type &>;
    { t[std::declval<typename T::index_type>()] } -> std::convertible_to<const typename T::value_type &>;
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
    { T::error_checking() } -> std::same_as<error_checking>;
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
concept fixed_shape =
    tensorial<T> && is_index_sequence<typename T::shape_type>::value && requires(T t, std::size_t index) {
        { t.template subview<0, 0>(index, index) };
    };

/**
 * @concept dynamic_shape
 * @brief Concept for tensors with a dynamic shape.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept dynamic_shape =
    tensorial<T> && std::is_same_v<typename T::shape_type, std::vector<std::size_t>> &&
    requires(T t, const std::vector<std::size_t> &subview_shape, const std::vector<std::size_t> &start_indices) {
        { t.subview(subview_shape, start_indices) } -> std::same_as<decltype(t.subview(subview_shape, start_indices))>;
    };

/**
 * @concept const_tensor
 * @brief Concept for constant tensors.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept const_tensor = std::is_const_v<std::remove_reference_t<T>>;

} // namespace squint

#endif // SQUINT_CORE_CONCEPTS_HPP