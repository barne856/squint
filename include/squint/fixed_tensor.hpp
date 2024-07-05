#ifndef SQUINT_FIXED_TENSOR_HPP
#define SQUINT_FIXED_TENSOR_HPP

#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <array>

namespace squint {

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

} // namespace squint

#endif // SQUINT_FIXED_TENSOR_HPP