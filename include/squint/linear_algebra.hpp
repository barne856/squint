#ifndef SQUINT_LINEAR_ALGEBRA_HPP
#define SQUINT_LINEAR_ALGEBRA_HPP

#include "squint/dynamic_tensor.hpp"
#include "squint/fixed_tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <algorithm>

namespace squint {

// Linear algebra mixin
template <typename Derived> class linear_algebra_mixin {
  public:
    // Element-wise addition
    template <tensor OtherDerived> constexpr auto operator+(const linear_algebra_mixin<OtherDerived> &other) const {
        const auto &a = static_cast<const Derived &>(*this);
        const auto &b = static_cast<const OtherDerived &>(other);

        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Incompatible shapes for addition");
        }

        if constexpr (fixed_shape_tensor<Derived> && fixed_shape_tensor<OtherDerived>) {
            constexpr auto shape = Derived::constexpr_shape();
            fixed_tensor<typename Derived::value_type, a.get_layout(), shape[0], shape[1]> result;
            for (std::size_t i = 0; i < a.size(); ++i) {
                result.at(i) = a.at(i) + b.at(i);
            }
            return result;
        } else {
            dynamic_tensor<typename Derived::value_type> result(a.shape());
            for (std::size_t i = 0; i < a.size(); ++i) {
                result.at(i) = a.at(i) + b.at(i);
            }
            return result;
        }
    }

    // Scalar multiplication
    constexpr auto operator*(const typename Derived::value_type &scalar) const {
        const auto &a = static_cast<const Derived &>(*this);

        if constexpr (fixed_shape_tensor<Derived>) {
            constexpr auto shape = Derived::constexpr_shape();
            fixed_tensor<typename Derived::value_type, a.get_layout(), shape[0], shape[1]> result;
            for (std::size_t i = 0; i < a.size(); ++i) {
                result.at(i) = a.at(i) * scalar;
            }
            return result;
        } else {
            dynamic_tensor<typename Derived::value_type> result(a.shape());
            for (std::size_t i = 0; i < a.size(); ++i) {
                result.at(i) = a.at(i) * scalar;
            }
            return result;
        }
    }
};

// Fixed tensor with linear algebra
template <typename T, layout L, std::size_t... Dims>
class fixed_tensor_with_la : public fixed_tensor<T, L, Dims...>,
                             public linear_algebra_mixin<fixed_tensor_with_la<T, L, Dims...>> {
  public:
    using fixed_tensor<T, L, Dims...>::fixed_tensor;

    // Add any additional methods or overrides specific to fixed_tensor_with_la
};

// Dynamic tensor with linear algebra
template <typename T>
class dynamic_tensor_with_la : public dynamic_tensor<T>, public linear_algebra_mixin<dynamic_tensor_with_la<T>> {
  public:
    using dynamic_tensor<T>::dynamic_tensor;

    // Add any additional methods or overrides specific to dynamic_tensor_with_la
};

// Deduction guide
template <typename T, typename... Args> dynamic_tensor_with_la(T, Args...) -> dynamic_tensor_with_la<T>;

} // namespace squint

#endif // SQUINT_LINEAR_ALGEBRA_HPP