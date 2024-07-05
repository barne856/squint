#ifndef SQUINT_TENSOR_VIEW_HPP
#define SQUINT_TENSOR_VIEW_HPP

#include "squint/tensor_base.hpp"
#include <array>
#include <vector>
#include <numeric>

namespace squint {

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
};

// View class for fixed_tensor
template <typename T, layout L, std::size_t... Dims>
class fixed_tensor_view : public tensor_view_base<fixed_tensor_view<T, L, Dims...>, T> {
  public:
    fixed_tensor_view(T *data, std::array<std::size_t, sizeof...(Dims)> strides)
        : tensor_view_base<fixed_tensor_view, T>(data, std::vector<std::size_t>(strides.begin(), strides.end()), L) {}

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return (Dims * ...); }
    static constexpr auto constexpr_shape() { return std::array<std::size_t, sizeof...(Dims)>{Dims...}; }
    constexpr std::vector<std::size_t> shape() const { return std::vector<std::size_t>{Dims...}; }

    template <typename... Indices>
    constexpr const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        return this->data_[calculate_offset(std::index_sequence_for<Indices...>{}, indices...)];
    }

  private:
    template <std::size_t... Is, typename... Indices>
    constexpr std::size_t calculate_offset(std::index_sequence<Is...> /*unused*/, Indices... indices) const {
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

} // namespace squint

#endif // SQUINT_TENSOR_VIEW_HPP