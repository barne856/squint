/**
 * @file tensor.hpp
 * @author Brendan Barnes
 * @brief Implementation of a tensor class using quantities
 */

#ifndef SQUINT_TENSOR_HPP
#define SQUINT_TENSOR_HPP

#include "squint/quantity.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace squint {

// Tensor concept
template <typename T>
concept tensor = requires(T t) {
    typename T::value_type;
    { t.rank() } -> std::convertible_to<std::size_t>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::same_as<std::vector<std::size_t>>;
    { t[0] } -> std::convertible_to<typename T::value_type>;
};

// Layout options
enum class layout { row_major, column_major };

// Forward declarations
template <typename T> class tensor_view;
template <typename T, layout L, std::size_t... Dims> class fixed_tensor;
template <typename T> class dynamic_tensor;

// Concepts
template<typename T>
concept has_constexpr_shape = requires(T t) {
    { T::constexpr_shape() } -> std::same_as<std::array<std::size_t, T::rank()>>;
};

template<typename T>
concept fixed_shape_tensor = tensor<T> && has_constexpr_shape<T>;

template<typename T>
concept dynamic_shape_tensor = tensor<T> && !has_constexpr_shape<T>;

// Linear algebra mixin
template <typename Derived>
class linear_algebra_mixin {
public:
    // Transpose method
    constexpr auto transpose() const {
        const auto& derived = static_cast<const Derived&>(*this);
        auto shape = derived.shape();
        std::reverse(shape.begin(), shape.end());

        if constexpr (fixed_shape_tensor<Derived>) {
            constexpr auto original_shape = Derived::constexpr_shape();
            return fixed_tensor_view<typename Derived::value_type, derived.get_layout(), original_shape[1], original_shape[0]>(
                const_cast<typename Derived::value_type*>(derived.data()),
                calculate_transposed_strides(original_shape)
            );
        } else {
            std::vector<std::size_t> strides = calculate_transposed_strides(shape);
            return dynamic_tensor_view<typename Derived::value_type>(
                const_cast<typename Derived::value_type*>(derived.data()),
                std::move(shape),
                std::move(strides),
                derived.get_layout()
            );
        }
    }

    // Element-wise addition
    template <tensor OtherDerived>
    constexpr auto operator+(const linear_algebra_mixin<OtherDerived>& other) const {
        const auto& a = static_cast<const Derived&>(*this);
        const auto& b = static_cast<const OtherDerived&>(other);

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
    constexpr auto operator*(const typename Derived::value_type& scalar) const {
        const auto& a = static_cast<const Derived&>(*this);

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
    template<typename Shape>
    static auto calculate_transposed_strides(const Shape& shape) {
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

// Base tensor class using CRTP
template <typename Derived, typename T> class tensor_base {
  public:
    using value_type = T;

    constexpr std::size_t rank() const { return static_cast<const Derived *>(this)->rank(); }
    constexpr std::size_t size() const { return static_cast<const Derived *>(this)->size(); }
    constexpr std::vector<std::size_t> shape() const { return static_cast<const Derived *>(this)->shape(); }

    // Multidimensional subscript operator (C++23)
    template <typename... Indices> constexpr T &operator[](Indices... indices) {
        return static_cast<Derived *>(this)->at(indices...);
    }

    template <typename... Indices> constexpr const T &operator[](Indices... indices) const {
        return static_cast<const Derived *>(this)->at(indices...);
    }

    template <typename... Indices> constexpr T &at(Indices... indices) {
        return static_cast<Derived *>(this)->at(std::vector<size_t>{static_cast<size_t>(indices)...});
    }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        return static_cast<const Derived *>(this)->at(std::vector<size_t>{static_cast<size_t>(indices)...});
    }

    // Add overloads for at() that take a vector of indices
    constexpr T &at(const std::vector<size_t> &indices) { return static_cast<Derived *>(this)->at_impl(indices); }

    constexpr const T &at(const std::vector<size_t> &indices) const {
        return static_cast<const Derived *>(this)->at_impl(indices);
    }

    virtual T *data() = 0;
    virtual const T *data() const = 0;

    // Output stream operator
    friend std::ostream &operator<<(std::ostream &os, const tensor_base &tensor) {
        const Derived &derived = static_cast<const Derived &>(tensor);
        os << "Tensor(shape=[";
        const auto &shape = derived.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            os << shape[i];
            if (i < shape.size() - 1)
                os << ", ";
        }
        os << "], data=";

        // Helper function to recursively print tensor data
        std::function<void(size_t, const std::vector<size_t> &)> print_data = [&](size_t depth,
                                                                                  const std::vector<size_t> &indices) {
            if (depth == shape.size()) {
                os << derived.at(indices);
                return;
            }
            os << "[";
            for (size_t i = 0; i < shape[depth]; ++i) {
                std::vector<size_t> new_indices = indices;
                new_indices.push_back(i);
                print_data(depth + 1, new_indices);
                if (i < shape[depth] - 1)
                    os << ", ";
            }
            os << "]";
        };

        print_data(0, {});
        os << ")";
        return os;
    }
};

// Non-owning tensor view
// Base class for both types of views
template <typename Derived, typename T> class tensor_view_base : public tensor_base<Derived, T> {
  protected:
    T *data_;
    std::vector<std::size_t> strides_;
    layout layout_;

  public:
    tensor_view_base(T *data, std::vector<std::size_t> strides, layout l)
        : data_(data), strides_(std::move(strides)), layout_(l) {}

    T *data() override { return data_; }
    const T *data() const override { return data_; }

    layout get_layout() const { return layout_; }
    const std::vector<std::size_t> &strides() const { return strides_; }

    // Common view methods can go here
};

// View class for fixed_tensor
template <typename T, layout L, std::size_t... Dims>
class fixed_tensor_view : public tensor_view_base<fixed_tensor_view<T, L, Dims...>, T> {
    std::array<std::size_t, sizeof...(Dims)> shape_;

  public:
    fixed_tensor_view(T *data, std::array<std::size_t, sizeof...(Dims)> strides)
        : tensor_view_base<fixed_tensor_view, T>(data, std::vector<std::size_t>(strides.begin(), strides.end()), L),
          shape_{Dims...} {}

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return (Dims * ...); }
    constexpr std::vector<std::size_t> shape() const { return std::vector<std::size_t>(shape_.begin(), shape_.end()); }

    // Implement at() method
    template <typename... Indices> const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        return this->data_[calculate_offset(std::index_sequence_for<Indices...>{}, indices...)];
    }

  private:
    template <std::size_t... Is, typename... Indices>
    std::size_t calculate_offset(std::index_sequence<Is...> /*unused*/, Indices... indices) const {
        return ((indices * this->strides_[Is]) + ...);
    }
};

// View class for dynamic_tensor
template <typename T> class dynamic_tensor_view : public tensor_view_base<dynamic_tensor_view<T>, T> {
    std::vector<std::size_t> shape_;

  public:
    dynamic_tensor_view(T *data, std::vector<std::size_t> shape, std::vector<std::size_t> strides, layout l)
        : tensor_view_base<dynamic_tensor_view, T>(data, std::move(strides), l), shape_(std::move(shape)) {}

    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()); }
    const std::vector<std::size_t> &shape() const { return shape_; }

    // Implement at() method
    template <typename... Indices> const T &at(Indices... indices) const {
        if (sizeof...(Indices) != shape_.size()) {
            throw std::invalid_argument("Incorrect number of indices");
        }
        return this->data_[calculate_offset(std::vector<std::size_t>{static_cast<std::size_t>(indices)...})];
    }

  private:
    std::size_t calculate_offset(const std::vector<std::size_t> &indices) const {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * this->strides_[i];
        }
        return offset;
    }
};

// Fixed tensor implementation
template <typename T, layout L, std::size_t... Dims>
class fixed_tensor : public tensor_base<fixed_tensor<T, L, Dims...>, T> {
    static constexpr std::size_t total_size = (Dims * ...);
    std::array<T, total_size> data_;

  public:
    constexpr fixed_tensor() = default;

    // Constexpr constructor taking an array of elements
    constexpr fixed_tensor(const std::array<T, total_size> &elements) : data_(elements) {}

    // Copy constructor
    constexpr fixed_tensor(const fixed_tensor &) = default;

    // Move constructor
    constexpr fixed_tensor(fixed_tensor &&) noexcept = default;

    // Copy assignment
    constexpr fixed_tensor &operator=(const fixed_tensor &) = default;

    // Move assignment
    constexpr fixed_tensor &operator=(fixed_tensor &&) noexcept = default;

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return total_size; }
    static constexpr layout get_layout() { return L; }

    // Constexpr shape method returning std::array
    static constexpr auto constexpr_shape() { return std::array<std::size_t, sizeof...(Dims)>{Dims...}; }

    // Non-constexpr shape method returning std::vector to satisfy tensor concept
    constexpr std::vector<std::size_t> shape() const { return {Dims...}; }

    constexpr T &at_impl(const std::vector<size_t> &indices) { return data_[calculate_index(indices)]; }

    constexpr const T &at_impl(const std::vector<size_t> &indices) const { return data_[calculate_index(indices)]; }

    constexpr T *data() { return data_.data(); }

    constexpr const T *data() const { return data_.data(); }

    fixed_tensor_view<T, L, Dims...> view() {
        return fixed_tensor_view<T, L, Dims...>(this->data(), calculate_strides());
    }

    fixed_tensor_view<const T, L, Dims...> view() const {
        return fixed_tensor_view<const T, L, Dims...>(this->data(), calculate_strides());
    }

  private:
    constexpr size_t calculate_index(const std::vector<size_t> &indices) const {
        if (indices.size() != sizeof...(Dims)) {
            throw std::invalid_argument("Incorrect number of indices");
        }

        size_t index = 0;
        size_t stride = 1;
        constexpr std::array<size_t, sizeof...(Dims)> dims = {Dims...};

        if constexpr (L == layout::row_major) {
            for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
                index += indices[i] * stride;
                stride *= dims[i];
            }
        } else { // column_major
            for (size_t i = 0; i < sizeof...(Dims); ++i) {
                index += indices[i] * stride;
                stride *= dims[i];
            }
        }

        return index;
    }

    std::array<std::size_t, sizeof...(Dims)> calculate_strides() const {
        std::array<std::size_t, sizeof...(Dims)> strides;
        std::size_t stride = 1;
        for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= std::array<std::size_t, sizeof...(Dims)>{Dims...}[i];
        }
        return strides;
    }
};

// Dynamic tensor implementation
template <typename T> class dynamic_tensor : public tensor_base<dynamic_tensor<T>, T> {
    std::vector<T> data_;
    std::vector<std::size_t> shape_;
    layout layout_;

  public:
    dynamic_tensor(std::vector<std::size_t> shape, layout layout = layout::column_major)
        : shape_(std::move(shape)), layout_(layout) {
        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size);
    }

    // Copy constructor
    dynamic_tensor(const dynamic_tensor &) = default;

    // Move constructor
    dynamic_tensor(dynamic_tensor &&) noexcept = default;

    // Copy assignment
    dynamic_tensor &operator=(const dynamic_tensor &) = default;

    // Move assignment
    dynamic_tensor &operator=(dynamic_tensor &&) noexcept = default;

    constexpr std::size_t rank() const { return shape_.size(); }
    constexpr std::size_t size() const { return data_.size(); }
    constexpr std::vector<std::size_t> shape() const { return shape_; }
    constexpr layout get_layout() const { return layout_; }

    T &at_impl(const std::vector<size_t> &indices) { return data_[calculate_index(indices)]; }

    const T &at_impl(const std::vector<size_t> &indices) const { return data_[calculate_index(indices)]; }

    T *data() { return data_.data(); }

    const T *data() const { return data_.data(); }

    void reshape(std::vector<std::size_t> new_shape) {
        if (std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>()) != data_.size()) {
            throw std::invalid_argument("New shape must have the same total size");
        }
        shape_ = std::move(new_shape);
    }

    dynamic_tensor_view<T> view() {
        return dynamic_tensor_view<T>(this->data(), this->shape(), calculate_strides(), this->get_layout());
    }

    dynamic_tensor_view<const T> view() const {
        return dynamic_tensor_view<const T>(this->data(), this->shape(), calculate_strides(), this->get_layout());
    }

  private:
    size_t calculate_index(const std::vector<size_t> &indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Incorrect number of indices");
        }

        size_t index = 0;
        size_t stride = 1;

        if (layout_ == layout::row_major) {
            for (int i = shape_.size() - 1; i >= 0; --i) {
                index += indices[i] * stride;
                stride *= shape_[i];
            }
        } else { // column_major
            for (size_t i = 0; i < shape_.size(); ++i) {
                index += indices[i] * stride;
                stride *= shape_[i];
            }
        }

        return index;
    }
    std::vector<std::size_t> calculate_strides() const {
        std::vector<std::size_t> strides(this->shape().size());
        std::size_t stride = 1;
        if (this->get_layout() == layout::row_major) {
            for (int i = this->shape().size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= this->shape()[i];
            }
        } else {
            for (std::size_t i = 0; i < this->shape().size(); ++i) {
                strides[i] = stride;
                stride *= this->shape()[i];
            }
        }
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

// Deduction guides
template <typename T, typename... Args> dynamic_tensor(T, Args...) -> dynamic_tensor<T>;

template <typename T, typename... Args> dynamic_tensor_with_la(T, Args...) -> dynamic_tensor_with_la<T>;

} // namespace squint

#endif // SQUINT_TENSOR_HPP