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
    // Transpose method
    constexpr auto transpose() const {
        const auto &derived = static_cast<const Derived &>(*this);
        auto shape = derived.shape();
        std::reverse(shape.begin(), shape.end());

        if constexpr (fixed_shape_tensor<Derived>) {
            constexpr auto original_shape = Derived::constexpr_shape();
            return fixed_tensor_view<typename Derived::value_type, derived.get_layout(), original_shape[1],
                                     original_shape[0]>(const_cast<typename Derived::value_type *>(derived.data()),
                                                        calculate_transposed_strides(original_shape));
        } else {
            std::vector<std::size_t> strides = calculate_transposed_strides(shape);
            return dynamic_tensor_view<typename Derived::value_type>(
                const_cast<typename Derived::value_type *>(derived.data()), std::move(shape), std::move(strides),
                derived.get_layout());
        }
    }

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

  private:
    // Helper function to calculate transposed strides
    template <typename Shape> static auto calculate_transposed_strides(const Shape &shape) {
        std::vector<std::size_t> strides(shape.size());
        std::size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        std::reverse(strides.begin(), strides.end());
        return strides;
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