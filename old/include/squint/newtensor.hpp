#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "squint/dimension.hpp"

namespace squint {

template <typename T, dimensional D> class quantity {
  public:
    using value_type = T;     // data type of the element
    using dimension_type = D; // dimension of the quantity
    constexpr quantity() = default;
};


template <typename T> struct quantity_traits {
    using dimension_type = dimensions::dimensionless;
    static constexpr bool is_quantity = false;
};

template <typename T, dimensional D> struct quantity_traits<quantity<T, D>> {
    using dimension_type = D;
    static constexpr bool is_quantity = true;
};

template <typename T>
concept quantitative = quantity_traits<T>::is_quantity;

template <class Derived, typename T, typename ShapeType> class tensor_base_crtp {
  protected:
    constexpr Derived &derived() { return static_cast<Derived &>(*this); }
    constexpr const Derived &derived() const { return static_cast<const Derived &>(*this); }

  public:
    using value_type = T;
    using dimension_type = typename quantity_traits<T>::dimension_type;

    constexpr const ShapeType &shape() const { return derived().shape_impl(); }
    constexpr size_t size() const { return derived().size_impl(); }

    // Common methods that can use derived().method_impl() to access specific implementations
    constexpr auto transpose() const { return derived().transpose_impl(); }

    constexpr auto flatten() const { return derived().flatten_impl(); }

    // Other common methods...

    template <typename U> auto operator+(const tensor_base_crtp<Derived, U, ShapeType> &other) const {
        using result_type = decltype(std::declval<T>() + std::declval<U>());
        return derived().template binary_op_impl<result_type>(other, std::plus<>());
    }

    template <typename U> auto operator*(const tensor_base_crtp<Derived, U, ShapeType> &other) const {
        using result_type = decltype(std::declval<T>() * std::declval<U>());
        return derived().template binary_op_impl<result_type>(other, std::multiplies<>());
    }

    // Specific implementations of common methods

    template <typename U = Derived> constexpr auto transpose_impl() const {
        if constexpr (std::is_same_v<U, tensor<T, sizes...>>) {
            // Fixed-size implementation
        } else {
            // Dynamic-size implementation
        }
    }

    // Other specialized implementations...
};

// Fixed-size tensor base
template <class Derived, typename T, int... sizes>
class fixed_tensor_base : public tensor_base_crtp<Derived, T, std::array<size_t, sizeof...(sizes)>> {
  protected:
    static constexpr std::array<size_t, sizeof...(sizes)> shape_array = {static_cast<size_t>(sizes)...};

  public:
    constexpr const auto &shape_impl() const { return shape_array; }
    static constexpr size_t size_impl() { return (... * sizes); }

    // Implement fixed-size specific methods...
};

// Dynamic tensor base
template <class Derived, typename T>
class dynamic_tensor_base : public tensor_base_crtp<Derived, T, std::vector<size_t>> {
  protected:
    std::vector<size_t> _shape;

  public:
    const auto &shape_impl() const { return _shape; }
    size_t size_impl() const { return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>()); }

    // Implement dynamic-size specific methods...
};

template <typename T, int... sizes> class tensor : public fixed_tensor_base<tensor<T, sizes...>, T, sizes...> {
    T _data[sizeof...(sizes)];

  public:
    constexpr T *data() { return _data; }
    constexpr const T *data() const { return _data; }

    // Implement required _impl methods...
};

template <typename T> class dynamic_tensor : public dynamic_tensor_base<dynamic_tensor<T>, T> {
    std::vector<T> _data;

  public:
    T *data() { return _data.data(); }
    const T *data() const { return _data.data(); }

    // Implement required _impl methods...
};

template <typename Base> class linear_algebra_mixin : public Base {
  public:
    constexpr auto matrix_multiply(const Base &other) const {
        // Implementation using Base::data() and Base::shape()
    }

    // Other linear algebra operations...
};

// Mixins for fixed-size and dynamic-size tensors
template <typename T, int... sizes> using linear_algebra_tensor = linear_algebra_mixin<tensor<T, sizes...>>;

// Tensor type traits and concepts
template <typename T> struct is_fixed_tensor : std::false_type {};

template <typename T, int... sizes> struct is_fixed_tensor<tensor<T, sizes...>> : std::true_type {};

template <typename T>
concept FixedTensor = is_fixed_tensor<T>::value;

template <typename T>
concept DynamicTensor = !FixedTensor<T> && requires(T t) {
    { t.shape() } -> std::convertible_to<std::vector<size_t>>;
};

template <FixedTensor T> auto specialized_operation(const T &t) {
    // Implementation for fixed-size tensors
}

template <DynamicTensor T> auto specialized_operation(const T &t) {
    // Implementation for dynamic-size tensors
}
} // namespace squint