#ifndef SQUINT_TENSOR_TENSOR_HPP
#define SQUINT_TENSOR_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/util/array_utils.hpp"

#include <array>
#include <vector>
#include <stdexcept>
#include <numeric>

namespace squint {

template <typename T, typename Shape, layout Layout = layout::row_major, error_checking ErrorChecking = error_checking::disabled>
class tensor {
public:
    using value_type = T;
    using shape_type = Shape;

    static constexpr auto layout() -> layout { return Layout; }
    static constexpr auto error_checking() -> error_checking { return ErrorChecking; }

    tensor() = default;

    template <typename... Dims>
    explicit tensor(Dims... dims) 
        requires (sizeof...(Dims) > 0 && (std::is_convertible_v<Dims, std::size_t> && ...))
    {
        if constexpr (is_index_sequence<Shape>::value) {
            static_assert(sizeof...(Dims) == Shape::size(), "Number of dimensions must match Shape");
            shape_ = std::array<std::size_t, Shape::size()>{static_cast<std::size_t>(dims)...};
        } else {
            shape_ = {static_cast<std::size_t>(dims)...};
        }
        allocate_and_initialize();
    }

    [[nodiscard]] constexpr auto rank() const -> std::size_t {
        if constexpr (is_index_sequence<Shape>::value) {
            return Shape::size();
        } else {
            return shape_.size();
        }
    }

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        if constexpr (is_index_sequence<Shape>::value) {
            return product(Shape{});
        } else {
            return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<std::size_t>());
        }
    }

    auto shape() const -> const auto& {
        return shape_;
    }

    auto strides() const -> const auto& {
        return strides_;
    }

    auto data() const -> const T* { return data_.data(); }
    auto data() -> T* { return data_.data(); }

    template <typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) == rank() || !is_index_sequence<Shape>::value,
                      "Number of indices must match tensor rank for fixed shape tensors");
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::array<std::size_t, sizeof...(Indices)>{static_cast<std::size_t>(indices)...});
        }
        return data_[compute_index(indices...)];
    }

    template <typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == rank() || !is_index_sequence<Shape>::value,
                      "Number of indices must match tensor rank for fixed shape tensors");
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(std::array<std::size_t, sizeof...(Indices)>{static_cast<std::size_t>(indices)...});
        }
        return data_[compute_index(indices...)];
    }

private:
    std::conditional_t<is_index_sequence<Shape>::value, 
                       std::array<std::size_t, Shape::size()>, 
                       std::vector<std::size_t>> shape_;
    std::conditional_t<is_index_sequence<Shape>::value, 
                       std::array<std::size_t, Shape::size()>, 
                       std::vector<std::size_t>> strides_;
    std::vector<T> data_;

    void allocate_and_initialize() {
        data_.resize(size());
        calculate_strides();
    }

    void calculate_strides() {
        if constexpr (is_index_sequence<Shape>::value) {
            strides_ = std::array<std::size_t, Shape::size()>{};
        } else {
            strides_.resize(rank());
        }

        if constexpr (Layout == layout::row_major) {
            strides_.back() = 1;
            for (int i = rank() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        } else {
            strides_.front() = 1;
            for (size_t i = 1; i < rank(); ++i) {
                strides_[i] = strides_[i - 1] * shape_[i - 1];
            }
        }
    }

    template <typename... Indices>
    std::size_t compute_index(Indices... indices) const {
        std::size_t index = 0;
        size_t i = 0;
        ((index += static_cast<std::size_t>(indices) * strides_[i++]), ...);
        return index;
    }

    template <typename Indices>
    void check_bounds(const Indices& indices) const {
        for (size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_HPP