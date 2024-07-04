#ifndef SQUINT_TENSOR_HPP
#define SQUINT_TENSOR_HPP
#include "squint/quantity.hpp"
#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <vector>

namespace squint {
// Concept for checking if a type is a tensor
template<typename t>
concept tensor = requires(const t& a) {
    typename t::value_type;
    typename t::shape_type;
    { a.shape() } -> std::convertible_to<typename t::shape_type>;
    { a.size() } -> std::convertible_to<size_t>;
};

// Base class for all tensors using CRTP
template<typename derived, typename t, typename shape_type>
class tensor_base {
protected:
    derived& as_derived() { return static_cast<derived&>(*this); }
    const derived& as_derived() const { return static_cast<const derived&>(*this); }

public:
    using value_type = t;
    using dimension_type = typename quantity_traits<value_type>::dimension_type;

    const shape_type& shape() const { return as_derived().shape_impl(); }
    size_t size() const { return as_derived().size_impl(); }

    // Binary operation implementation
    template<typename u, typename op>
    auto binary_op(const tensor_base<derived, u, shape_type>& other, op operation) const {
        using result_type = decltype(operation(std::declval<value_type>(), std::declval<u>()));
        return as_derived().template binary_op_impl<result_type>(other.as_derived(), operation);
    }

    // Arithmetic operations
    template<tensor u>
    auto operator+(const u& other) const {
        return binary_op(other, std::plus<>());
    }

    template<tensor u>
    auto operator-(const u& other) const {
        return binary_op(other, std::minus<>());
    }
};

// fixed_tensor implementation
template<typename t, size_t... sizes>
class fixed_tensor : public tensor_base<fixed_tensor<t, sizes...>, t, std::array<size_t, sizeof...(sizes)>> {
    std::array<t, (sizes * ...)> data_{};

public:
    using value_type = t;
    using shape_type = std::array<size_t, sizeof...(sizes)>;

    fixed_tensor() = default;

    static constexpr shape_type shape() { return {sizes...}; }
    static constexpr size_t size() { return (sizes * ...); }

    const auto& shape_impl() const { return shape(); }
    constexpr size_t size_impl() const { return size(); }

    template<typename result_type, typename u, typename op>
    auto binary_op_impl(const fixed_tensor<u, sizes...>& other, op operation) const {
        fixed_tensor<result_type, sizes...> result;
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = operation(data_[i], other.data_[i]);
        }
        return result;
    }
};

// dynamic_tensor implementation
template<typename t>
class dynamic_tensor : public tensor_base<dynamic_tensor<t>, t, std::vector<size_t>> {
    std::vector<t> data_;
    std::vector<size_t> shape_;

public:
    using value_type = t;
    using shape_type = std::vector<size_t>;

    dynamic_tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_size = 1;
        for (size_t s : shape) total_size *= s;
        data_.resize(total_size);
    }

    const shape_type& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    const auto& shape_impl() const { return shape_; }
    size_t size_impl() const { return data_.size(); }

    template<typename result_type, typename u, typename op>
    auto binary_op_impl(const dynamic_tensor<u>& other, op operation) const {
        dynamic_tensor<result_type> result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = operation(data_[i], other.data_[i]);
        }
        return result;
    }
};

// linear_algebra mixin
template<tensor tensor_type>
class linear_algebra : public tensor_type {
public:
    using tensor_type::tensor_type;

    // Matrix multiplication
    template<tensor other_tensor>
    auto matmul(const other_tensor& other) const {
        // static_assert(tensor_type::shape_type::size() == 2 && other_tensor::shape_type::size() == 2,
        //               "matmul is only implemented for 2D matrices");

        using result_value_type = decltype(std::declval<typename tensor_type::value_type>() * 
                                           std::declval<typename other_tensor::value_type>());
        using result_dimension = mult_t<typename tensor_type::dimension_type, 
                                        typename other_tensor::dimension_type>;
        using result_quantity = quantity<result_value_type, result_dimension>;

        const auto& shape1 = this->shape();
        const auto& shape2 = other.shape();
        assert(shape1[1] == shape2[0] && "Incompatible matrix dimensions for multiplication");

        if constexpr (std::is_same_v<typename tensor_type::shape_type, std::array<size_t, 2>>) {
            // Fixed tensor
            constexpr size_t rows = std::tuple_size_v<typename tensor_type::shape_type>;
            constexpr size_t cols = std::tuple_size_v<typename other_tensor::shape_type>;
            return fixed_tensor<result_quantity, rows, cols>();
        } else {
            // Dynamic tensor
            return dynamic_tensor<result_quantity>({shape1[0], shape2[1]});
        }
    }
};

// Convenience type aliases
template<typename t, size_t... sizes>
using la_fixed_tensor = linear_algebra<fixed_tensor<t, sizes...>>;

template<typename t>
using la_dynamic_tensor = linear_algebra<dynamic_tensor<t>>;

}
#endif