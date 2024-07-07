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
    constexpr fixed_tensor(const std::array<T, total_size> &elements) : data_(elements) {}
    constexpr fixed_tensor(const fixed_tensor &) = default;
    constexpr fixed_tensor(fixed_tensor &&) noexcept = default;
    constexpr fixed_tensor &operator=(const fixed_tensor &) = default;
    constexpr fixed_tensor &operator=(fixed_tensor &&) noexcept = default;

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return total_size; }
    static constexpr layout get_layout() { return L; }
    static constexpr auto constexpr_shape() { return std::array<std::size_t, sizeof...(Dims)>{Dims...}; }
    constexpr std::vector<std::size_t> shape() const { return {Dims...}; }

    template <typename... Indices> constexpr T &at(Indices... indices) {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
    }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
    }

    // Non-constexpr version for runtime index access
    T &at_impl(const std::vector<size_t> &indices) {
        if (indices.size() != sizeof...(Dims)) {
            throw std::invalid_argument("Incorrect number of indices");
        }
        return data_[calculate_index_runtime(indices)];
    }

    const T &at_impl(const std::vector<size_t> &indices) const {
        if (indices.size() != sizeof...(Dims)) {
            throw std::invalid_argument("Incorrect number of indices");
        }
        return data_[calculate_index_runtime(indices)];
    }

    constexpr T *data() { return data_.data(); }
    constexpr const T *data() const { return data_.data(); }

    constexpr fixed_tensor_view<T, L, Dims...> view() {
        return fixed_tensor_view<T, L, Dims...>(this->data(), calculate_strides());
    }

    constexpr fixed_tensor_view<const T, L, Dims...> view() const {
        return fixed_tensor_view<const T, L, Dims...>(this->data(), calculate_strides());
    }

  private:
    template <std::size_t... Is, typename... Indices>
    static constexpr size_t calculate_index(std::index_sequence<Is...> /*unused*/, Indices... indices) {
        constexpr std::array<size_t, sizeof...(Dims)> dims = {Dims...};
        size_t index = 0;
        size_t stride = 1;
        ((index += (L == layout::row_major ? (sizeof...(Dims) - 1 - Is) : Is) < sizeof...(Dims) - 1 ? indices * stride
                                                                                                    : indices * stride,
          stride *= dims[L == layout::row_major ? sizeof...(Dims) - 1 - Is : Is]),
         ...);
        return index;
    }

    // Non-constexpr version for runtime index calculation
    size_t calculate_index_runtime(const std::vector<size_t> &indices) const {
        constexpr std::array<size_t, sizeof...(Dims)> dims = {Dims...};
        size_t index = 0;
        size_t stride = 1;
        if constexpr (L == layout::row_major) {
            for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
                index += indices[i] * stride;
                stride *= dims[i];
            }
        } else {
            for (size_t i = 0; i < sizeof...(Dims); ++i) {
                index += indices[i] * stride;
                stride *= dims[i];
            }
        }
        return index;
    }

    static constexpr std::array<std::size_t, sizeof...(Dims)> calculate_strides() {
        std::array<std::size_t, sizeof...(Dims)> strides;
        std::size_t stride = 1;
        constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
        if constexpr (L == layout::row_major) {
            for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= dims[i];
            }
        } else { // column_major
            for (size_t i = 0; i < sizeof...(Dims); ++i) {
                strides[i] = stride;
                stride *= dims[i];
            }
        }
        return strides;
    }
};

} // namespace squint

#endif // SQUINT_FIXED_TENSOR_HPP