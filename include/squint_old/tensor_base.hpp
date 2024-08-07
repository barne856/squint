#ifndef SQUINT_TENSOR_BASE_HPP
#define SQUINT_TENSOR_BASE_HPP

#include "squint/core.hpp"
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

namespace squint {

// Forward declarations
template <typename T, layout L, error_checking ErrorChecking, std::size_t... Dims> class fixed_tensor;
template <typename T, error_checking ErrorChecking> class dynamic_tensor;
template <typename T, layout L, typename Strides, error_checking ErrorChecking, std::size_t... Dims>
class fixed_tensor_view;
template <typename T, layout L, typename Strides, error_checking ErrorChecking, std::size_t... Dims>
class const_fixed_tensor_view;
template <typename T, error_checking ErrorChecking> class dynamic_tensor_view;
template <typename T, error_checking ErrorChecking> class const_dynamic_tensor_view;

// Tensor concept
template <typename T>
concept tensor = requires(T t) {
    typename T::value_type;
    { t.rank() } -> std::convertible_to<std::size_t>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::same_as<std::vector<std::size_t>>;
    { t.get_layout() } -> std::same_as<layout>;
    { t.strides() } -> std::convertible_to<std::vector<std::size_t>>;
    { t[0] } -> std::convertible_to<typename T::value_type>;
};

template <typename T>
concept has_constexpr_shape = requires(T t) {
    { T::constexpr_shape() } -> std::same_as<std::array<std::size_t, T::rank()>>;
};

// has constexpr strides
template <typename T>
concept has_constexpr_strides = requires(T t) {
    { T::constexpr_strides() } -> std::same_as<std::array<std::size_t, T::rank()>>;
};

template <typename T>
concept fixed_shape_tensor = tensor<T> && has_constexpr_shape<T> && has_constexpr_strides<T>;

template <typename T>
concept dynamic_shape_tensor = tensor<T> && !has_constexpr_shape<T> && !has_constexpr_strides<T>;

// Base tensor class using CRTP
template <typename Derived, typename T, error_checking ErrorChecking> class __declspec(empty_bases) tensor_base {
  public:
    using value_type = T;

    constexpr std::size_t rank() const { return static_cast<const Derived *>(this)->rank(); }
    constexpr std::size_t size() const { return static_cast<const Derived *>(this)->size(); }
    constexpr std::vector<std::size_t> shape() const { return static_cast<const Derived *>(this)->shape(); }
    constexpr std::vector<std::size_t> strides() const { return static_cast<const Derived *>(this)->strides(); }
    constexpr layout get_layout() const { return static_cast<const Derived *>(this)->get_layout(); }

    template <typename... Indices> constexpr T &operator()(Indices... indices) {
        return static_cast<Derived *>(this)->at(indices...);
    }
    template <typename... Indices> constexpr const T &operator()(Indices... indices) const {
        return static_cast<const Derived *>(this)->at(indices...);
    }
    constexpr T &operator[](std::size_t index) {
        return static_cast<Derived *>(this)->at(index);
    }
    constexpr const T &operator[](std::size_t index) const {
        return static_cast<const Derived *>(this)->at(index);
    }
    #ifndef _MSC_VER
    // Multidimensional subscript operator (C++23) MSVC does not support this yet
    template <typename... Indices> constexpr T &operator[](Indices... indices) {
        return static_cast<Derived *>(this)->at(indices...);
    }
    template <typename... Indices> constexpr const T &operator[](Indices... indices) const {
        return static_cast<const Derived *>(this)->at(indices...);
    }
    #endif
    template <typename... Indices> constexpr T &at(Indices... indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return static_cast<Derived *>(this)->at(std::vector<size_t>{static_cast<size_t>(indices)...});
    }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return static_cast<const Derived *>(this)->at(std::vector<size_t>{static_cast<size_t>(indices)...});
    }

    // Add overloads for at() that take a vector of indices
    constexpr T &at(const std::vector<size_t> &indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(indices);
        }
        return static_cast<Derived *>(this)->at_impl(indices);
    }

    constexpr const T &at(const std::vector<size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(indices);
        }
        return static_cast<const Derived *>(this)->at_impl(indices);
    }

    void check_bounds(const std::vector<size_t> &indices) const {
        const auto &shape = this->shape();
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Invalid number of indices");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    void check_subview_bounds(const std::vector<std::size_t> &shape, const std::vector<std::size_t>& start) const {
        if (shape.size() != start.size() || shape.size() != this->shape().size()) {
            throw std::out_of_range("Invalid number of sizes or offsets");
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (start[i] + shape[i] > this->shape()[i]) {
                throw std::out_of_range("Subview out of bounds");
            }
        }
    }

    T *data() { return static_cast<Derived *>(this)->data(); }
    const T *data() const { return static_cast<const Derived *>(this)->data(); }

    // Helper function to recursively unpack indices for fixed tensors
    template <size_t... Is>
    static constexpr auto unpack_indices(const std::vector<size_t> &indices, std::index_sequence<Is...> /*unused*/) {
        return [&](const auto &derived) { return derived.at(indices[Is]...); };
    }

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
        std::function<void(size_t, std::vector<size_t> &)> print_data = [&](size_t depth,
                                                                            std::vector<size_t> &indices) {
            if (depth == shape.size()) {
                if constexpr (fixed_shape_tensor<Derived>) {
                    constexpr size_t rank = Derived::rank();
                    os << unpack_indices(indices, std::make_index_sequence<rank>{})(derived);
                } else {
                    os << derived.at_impl(indices);
                }
                return;
            }
            os << "[";
            for (size_t i = 0; i < shape[depth]; ++i) {
                indices[depth] = i;
                print_data(depth + 1, indices);
                if (i < shape[depth] - 1)
                    os << ", ";
            }
            os << "]";
        };

        std::vector<size_t> indices(shape.size(), 0);
        print_data(0, indices);
        os << ")";
        return os;
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_BASE_HPP