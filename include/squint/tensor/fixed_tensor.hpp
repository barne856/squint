#ifndef SQUINT_TENSOR_FIXED_TENSOR_HPP
#define SQUINT_TENSOR_FIXED_TENSOR_HPP

#include "squint/tensor/iterable_tensor.hpp"
#include "squint/util/array_utils.hpp"
#include <array>
#include <initializer_list>
#include <random>

namespace squint {

template <typename T, typename Shape, layout Layout = layout::row_major,
          error_checking ErrorChecking = error_checking::disabled>
class fixed_tensor
    : public iterable_tensor<fixed_tensor<T, Shape, Layout, ErrorChecking>, T, Shape, Layout, ErrorChecking> {
  public:
    using value_type = T;
    using shape_type = Shape;

    // Required methods for tensor_base
    static constexpr std::size_t rank() { return Shape::size(); }
    static constexpr std::size_t size() { return product(Shape{}); }

    std::vector<std::size_t> shape() const {
        constexpr auto shape_arr = make_array(Shape{});
        return std::vector<std::size_t>(shape_arr.begin(), shape_arr.end());
    }

    std::vector<std::size_t> strides() const {
        constexpr auto shape_arr = make_array(Shape{});
        std::vector<std::size_t> strides(rank());
        if constexpr (Layout == layout::row_major) {
            strides[rank() - 1] = 1;
            for (std::size_t i = rank() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape_arr[i];
            }
        } else {
            strides[0] = 1;
            for (std::size_t i = 0; i < rank() - 1; ++i) {
                strides[i + 1] = strides[i] * shape_arr[i];
            }
        }
        return strides;
    }

    T *data() { return data_.data(); }
    const T *data() const { return data_.data(); }

    static constexpr layout layout() { return Layout; }
    static constexpr error_checking error_checking() { return ErrorChecking; }

    // Constructors
    fixed_tensor() = default;

    explicit fixed_tensor(const std::array<T, size()> &data) : data_(data) {}

    fixed_tensor(std::initializer_list<T> init) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (init.size() != size()) {
                throw std::invalid_argument("Initializer list size does not match tensor size");
            }
        }
        std::copy(init.begin(), init.end(), data_.begin());
    }

    explicit fixed_tensor(const T &value) { data_.fill(value); }

    // Static methods
    static fixed_tensor zeros() { return fixed_tensor(T(0)); }

    static fixed_tensor ones() { return fixed_tensor(T(1)); }

    static fixed_tensor full(const T &value) { return fixed_tensor(value); }

    static fixed_tensor arange(T start = T(0), T step = T(1)) {
        fixed_tensor result;
        T value = start;
        for (auto &elem : result.data_) {
            elem = value;
            value += step;
        }
        return result;
    }

    static fixed_tensor diag(const std::array<T, min(Shape{})> &diagonal) {
        fixed_tensor result;
        constexpr auto shape_arr = make_array(Shape{});
        constexpr std::size_t min_dim = min(Shape{});
        for (std::size_t i = 0; i < min_dim; ++i) {
            result.data_[i * (size() / shape_arr[0] + 1)] = diagonal[i];
        }
        return result;
    }

    static fixed_tensor random(T min = T(0), T max = T(1)) {
        fixed_tensor result;
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(min, max);
            for (auto &elem : result.data_) {
                elem = dis(gen);
            }
        } else {
            std::uniform_real_distribution<T> dis(min, max);
            for (auto &elem : result.data_) {
                elem = dis(gen);
            }
        }
        return result;
    }

    static fixed_tensor eye() {
        fixed_tensor result;
        constexpr std::size_t min_dim = min(Shape{});
        constexpr auto shape_arr = make_array(Shape{});
        for (std::size_t i = 0; i < min_dim; ++i) {
            result.data_[i * (size() / shape_arr[0] + 1)] = T(1);
        }
        return result;
    }

    // Element access
    template <typename... Indices> T &operator()(Indices... indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if constexpr (sizeof...(Indices) != rank()) {
                throw std::invalid_argument("Invalid number of indices");
            }
            check_bounds(std::array<std::size_t, rank()>{static_cast<std::size_t>(indices)...});
        }
        return data_[compute_index(indices...)];
    }

    template <typename... Indices> const T &operator()(Indices... indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if constexpr (sizeof...(Indices) != rank()) {
                throw std::invalid_argument("Invalid number of indices");
            }
            check_bounds(std::array<std::size_t, rank()>{static_cast<std::size_t>(indices)...});
        }
        return data_[compute_index(indices...)];
    }

    T &operator[](const std::array<std::size_t, rank()> &indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(indices);
        }
        return data_[compute_index(indices)];
    }

    const T &operator[](const std::array<std::size_t, rank()> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(indices);
        }
        return data_[compute_index(indices)];
    }

    // Shape manipulation
    template <typename NewShape> auto reshape() const {
        static_assert(NewShape::size() > 0, "New shape must have at least one dimension");
        static_assert(product(NewShape{}) == size(), "New shape must have the same number of elements");
        return fixed_tensor<T, NewShape, Layout, ErrorChecking>(data_);
    }

    auto view() {
        // Implementation details for view...
    }
    auto view() const {
        // Implementation details for const view...
    }

    template <std::size_t... Dims> auto subview(std::size_t offset) {
        constexpr auto shape_arr = make_array(Shape{});
        static_assert(sizeof...(Dims) == rank(), "Subview dimensions must match tensor rank()");
        static_assert((... && (Dims <= shape_arr[sizeof...(Dims) - 1])),
                      "Subview dimensions must not exceed tensor dimensions");
        // Implementation details for subview...
    }

    // Data manipulation
    void fill(const T &value) { data_.fill(value); }

    auto flatten() const { return data_; }

    // Substructure views
    auto rows() { /* Implementation for rows */ }
    auto rows() const { /* Implementation for const rows */ }

    auto row(std::size_t index) { /* Implementation for row */ }
    auto row(std::size_t index) const { /* Implementation for const row */ }

    auto cols() { /* Implementation for cols */ }
    auto cols() const { /* Implementation for const cols */ }

    auto col(std::size_t index) { /* Implementation for col */ }
    auto col(std::size_t index) const { /* Implementation for const col */ }

  private:
    std::array<T, size()> data_;

    template <typename... Indices> std::size_t compute_index(Indices... indices) const {
        if constexpr (Layout == layout::row_major) {
            return compute_row_major_index(std::make_index_sequence<rank()>{}, indices...);
        } else {
            return compute_column_major_index(std::make_index_sequence<rank()>{}, indices...);
        }
    }

    template <std::size_t... I, typename... Indices>
    std::size_t compute_row_major_index(std::index_sequence<I...>, Indices... indices) const {
        constexpr auto shape_arr = make_array(Shape{});
        return ((std::get<I>(std::forward_as_tuple(indices...)) * (... * shape_arr[I + 1])) + ...);
    }

    template <std::size_t... I, typename... Indices>
    std::size_t compute_column_major_index(std::index_sequence<I...>, Indices... indices) const {
        constexpr auto shape_arr = make_array(Shape{});
        return ((std::get<I>(std::forward_as_tuple(indices...)) * (... * shape_arr[rank() - I - 1])) + ...);
    }

    void check_bounds(const std::array<std::size_t, rank()> &indices) const {
        constexpr auto shape_arr = make_array(Shape{});
        for (std::size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape_arr[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_FIXED_TENSOR_HPP