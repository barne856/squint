#ifndef SQUINT_FIXED_TENSOR_HPP
#define SQUINT_FIXED_TENSOR_HPP

#include "squint/iterable_tensor.hpp"
#include "squint/linear_algebra.hpp"
// #include "squint/quantity.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <array>
#include <cstddef>
#include <random>

namespace squint {

namespace detail {
template <typename Tensor, typename BlockTensor>
constexpr auto make_subviews_iterator(const BlockTensor & /*unused*/, Tensor &tensor) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return tensor.template subviews<BlockTensor::constexpr_shape()[Is]...>();
    }(std::make_index_sequence<BlockTensor::constexpr_shape().size()>{});
}
} // namespace detail

// Fixed tensor implementation
template <typename T, layout L, error_checking ErrorChecking, std::size_t... Dims>
class fixed_tensor : public iterable_tensor<fixed_tensor<T, L, ErrorChecking, Dims...>, T, ErrorChecking>,
                     public fixed_linear_algebra_mixin<fixed_tensor<T, L, ErrorChecking, Dims...>, ErrorChecking> {
    static constexpr std::size_t total_size = (Dims * ...);
    std::array<T, total_size> data_;

  public:
    using value_type = T;
    using iterable_tensor<fixed_tensor<T, L, ErrorChecking, Dims...>, T, ErrorChecking>::subviews;
    // virtual destructor
    virtual ~fixed_tensor() = default;
    constexpr fixed_tensor() = default;
    // insert elements into the layout
    constexpr fixed_tensor(const std::array<T, total_size> &elements) : data_(elements) {}
    // Fill the tensor with a single value
    explicit constexpr fixed_tensor(const T &value) { data_.fill(value); }
    // Fill the tensor with a single block or view
    template <fixed_shape_tensor BlockTensor> constexpr fixed_tensor(const BlockTensor &block) {
        static_assert(check_dimensions<BlockTensor>(std::make_index_sequence<BlockTensor::rank()>{}),
                      "Each dimension of the block must be a multiple of the corresponding dimension of the tensor");
        copy_from_block(block);
    }

    // Create a tensor from a list of tensor blocks or views
    template <fixed_shape_tensor BlockTensor, std::size_t N>
    constexpr fixed_tensor(const std::array<BlockTensor, N> &blocks) {
        static_assert(check_dimensions<BlockTensor>(std::make_index_sequence<BlockTensor::rank()>{}),
                      "Each dimension of the block must be a multiple of the corresponding dimension of the tensor");
        // the total size of the tensor must be a multiple of the total size of the block
        static_assert(total_size % BlockTensor::size() == 0, "Total size must be a multiple of block size");
        copy_from_blocks(blocks);
    }
    constexpr fixed_tensor(const fixed_tensor &) = default;
    constexpr fixed_tensor(fixed_tensor &&) noexcept = default;
    constexpr fixed_tensor &operator=(const fixed_tensor &) = default;
    constexpr fixed_tensor &operator=(fixed_tensor &&) noexcept = default;

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return total_size; }
    static constexpr layout get_layout() { return L; }
    static constexpr error_checking get_error_checking() { return ErrorChecking; }
    static constexpr std::vector<std::size_t> strides() {
        auto strides_array = calculate_strides();
        return std::vector<std::size_t>(std::begin(strides_array), std::end(strides_array));
    }
    static constexpr auto constexpr_strides() { return calculate_strides(); }
    static constexpr auto constexpr_shape() { return std::array<std::size_t, sizeof...(Dims)>{Dims...}; }
    constexpr std::vector<std::size_t> shape() const { return {Dims...}; }

    template <typename... Indices> constexpr T &at(Indices... indices) {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        // specialization for 1D tensors
        if constexpr (sizeof...(Dims) == 1) {
            return data_[indices...];
        }
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
    }

    // specialization for 1D tensors
    constexpr T &at(size_t index) {
        static_assert(1 == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= fixed_tensor::constexpr_shape()[0]) {
                throw std::out_of_range("Index out of range");
            }
        }
        return data_[index];
    }

    // specialization for 2D tensors
    constexpr T &at(size_t row, size_t col) {
        static_assert(2 == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (row >= fixed_tensor::constexpr_shape()[0] || col >= fixed_tensor::constexpr_shape()[1]) {
                throw std::out_of_range("Index out of range");
            }
        }
        if constexpr (L == layout::row_major) {
            return data_[row * fixed_tensor::constexpr_shape()[1] + col];
        } else {
            return data_[col * fixed_tensor::constexpr_shape()[0] + row];
        }
    }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
    }

    // specialization for 1D tensors
    constexpr const T &at(size_t index) const {
        static_assert(1 == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= fixed_tensor::constexpr_shape()[0]) {
                throw std::out_of_range("Index out of range");
            }
        }
        return data_[index];
    }

    // specialization for 2D tensors
    constexpr const T &at(size_t row, size_t col) const {
        static_assert(2 == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (row >= fixed_tensor::constexpr_shape()[0] || col >= fixed_tensor::constexpr_shape()[1]) {
                throw std::out_of_range("Index out of range");
            }
        }
        if constexpr (L == layout::row_major) {
            return data_[row * fixed_tensor::constexpr_shape()[1] + col];
        } else {
            return data_[col * fixed_tensor::constexpr_shape()[0] + row];
        }
    }

    // Non-constexpr version for runtime index access
    T &at_impl(const std::vector<size_t> &indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (indices.size() != sizeof...(Dims)) {
                throw std::invalid_argument("Incorrect number of indices");
            }
            this->check_bounds(indices);
        }
        return data_[calculate_index_runtime(indices)];
    }

    const T &at_impl(const std::vector<size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (indices.size() != sizeof...(Dims)) {
                throw std::invalid_argument("Incorrect number of indices");
            }
            this->check_bounds(indices);
        }
        return data_[calculate_index_runtime(indices)];
    }

    T *data() { return data_.data(); }
    const T *data() const { return data_.data(); }
    auto raw_data() {
        if constexpr (quantitative<T>) {
            return &(data_[0].value());
        } else {
            return data();
        }
    }

    auto raw_data() const {
        if constexpr (quantitative<T>) {
            return &(std::as_const(data_[0]).value());
        } else {
            return data();
        }
    }

    template <std::size_t... NewDims> auto reshape() {
        static_assert((NewDims * ...) == total_size, "New shape must have the same total size");
        using initial_strides = compile_time_strides<L, NewDims...>;
        return fixed_tensor_view<T, L, initial_strides, ErrorChecking, NewDims...>(data());
    }

    template <std::size_t... NewDims> auto reshape() const {
        static_assert((NewDims * ...) == total_size, "New shape must have the same total size");
        using initial_strides = compile_time_strides<L, NewDims...>;
        return const_fixed_tensor_view<T, L, initial_strides, ErrorChecking, NewDims...>(data());
    }

    constexpr auto view() { return make_fixed_tensor_view(*this); }

    constexpr auto view() const { return make_fixed_tensor_view(*this); }

    template <std::size_t... NewDims, typename... Slices> auto subview(Slices... slices) {
        return view().template subview<NewDims...>(slices...);
    }

    template <std::size_t... NewDims, typename... Slices> auto subview(Slices... slices) const {
        return view().template subview<NewDims...>(slices...);
    }

    static constexpr fixed_tensor zeros() {
        fixed_tensor result;
        result.data_.fill(T{});
        return result;
    }

    static constexpr fixed_tensor ones() {
        fixed_tensor result;
        result.data_.fill(T{1});
        return result;
    }

    static constexpr fixed_tensor full(const T &value) {
        fixed_tensor result;
        result.data_.fill(value);
        return result;
    }

    static constexpr fixed_tensor arange(T start, T step = T{1}) {
        fixed_tensor result;
        T value = start;
        for (std::size_t i = 0; i < total_size; ++i) {
            result.data_[i] = value;
            value += step;
        }
        return result;
    }

    template <fixed_shape_tensor OtherTensor> static constexpr fixed_tensor diag(const OtherTensor &vector) {
        static_assert(OtherTensor::rank() == 1, "Diagonal vector must be 1D");
        static_assert(OtherTensor::size() == std::min(Dims...),
                      "Diagonal tensor size must be the minimum of the dimensions");
        fixed_tensor result;
        result.fill(T{}); // Fill with zeros

        constexpr std::size_t min_dim = std::min({Dims...});
        for (std::size_t i = 0; i < min_dim; ++i) {
            std::array<std::size_t, sizeof...(Dims)> indices;
            indices.fill(i);
            result.at_impl(std::vector<std::size_t>(indices.begin(), indices.end())) = vector.at(i);
        }

        return result;
    }

    static constexpr fixed_tensor diag(const T &value) {
        fixed_tensor result;
        result.fill(T{}); // Fill with zeros

        constexpr std::size_t min_dim = std::min({Dims...});
        for (std::size_t i = 0; i < min_dim; ++i) {
            std::array<std::size_t, sizeof...(Dims)> indices;
            indices.fill(i);
            result.at_impl(std::vector<std::size_t>(indices.begin(), indices.end())) = value;
        }

        return result;
    }

    static fixed_tensor random(T low = T{}, T high = T{1}) {
        fixed_tensor result;
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (quantitative<T>) {
            std::uniform_real_distribution<typename T::value_type> dis(static_cast<typename T::value_type>(low),
                                                                       static_cast<typename T::value_type>(high));

            for (std::size_t i = 0; i < total_size; ++i) {
                result.data_[i] = T{dis(gen)};
            }
        } else {
            std::uniform_real_distribution<T> dis(low, high);

            for (std::size_t i = 0; i < total_size; ++i) {
                result.data_[i] = dis(gen);
            }
        }

        return result;
    }

    static constexpr fixed_tensor I() {
        static_assert(((Dims == Dims) && ...), "All dimensions must be equal for identity tensor");
        fixed_tensor result;
        result.fill(T{}); // Fill with zeros

        constexpr std::size_t min_dim = std::min({Dims...});
        for (std::size_t i = 0; i < min_dim; ++i) {
            std::array<std::size_t, sizeof...(Dims)> indices;
            indices.fill(i);
            result.at_impl(std::vector<std::size_t>(indices.begin(), indices.end())) = T{1};
        }

        return result;
    }

    void fill(const T &value) { data_.fill(value); }
    auto flatten() { return view().flatten(); }
    auto flatten() const { return view().flatten(); }

    auto rows() {
        if constexpr (sizeof...(Dims) == 0) {
            // For 0D tensors
            return this->template subviews<>();
        } else {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subviews<1, std::get<Is + 1>(dims)...>();
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        }
    }

    auto rows() const {
        if constexpr (sizeof...(Dims) == 0) {
            // For 0D tensors
            return this->template subviews<>();
        } else {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subviews<1, std::get<Is + 1>(dims)...>();
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        }
    }

    auto row(std::size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::array{Dims...}[0]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        if constexpr (sizeof...(Dims) == 0) {
            // For 0D tensors
            return this->template subview<>();
        } else {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subview<1, std::get<Is + 1>(dims)...>(slice{index, 1}, slice{0, Is}...);
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        }
    }

    auto row(std::size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::array{Dims...}[0]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        if constexpr (sizeof...(Dims) == 0) {
            // For 0D tensors
            return this->template subview<>();
        } else {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subview<1, std::get<Is + 1>(dims)...>(slice{index, 1}, slice{0, Is}...);
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        }
    }

    auto cols() {
        if constexpr (sizeof...(Dims) > 1) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subviews<std::get<Is>(dims)..., 1>();
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        } else {
            // For 1D tensors
            return this->template subviews<Dims...>();
        }
    }

    auto cols() const {
        if constexpr (sizeof...(Dims) > 1) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subviews<std::get<Is>(dims)..., 1>();
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        } else {
            // For 1D tensors
            return this->template subviews<Dims...>();
        }
    }

    auto col(std::size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::array{Dims...}[sizeof...(Dims) - 1]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        if constexpr (sizeof...(Dims) > 1) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subview<std::get<Is>(dims)..., 1>(slice{0, Is}..., slice{index, 1});
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        } else {
            // For 1D tensors
            return this->template subview<Dims...>(slice{0, size()});
        }
    }

    auto col(std::size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::array{Dims...}[sizeof...(Dims) - 1]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        if constexpr (sizeof...(Dims) > 1) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                constexpr std::array<std::size_t, sizeof...(Dims)> dims = {Dims...};
                return this->template subview<std::get<Is>(dims)..., 1>(slice{0, Is}..., slice{index, 1});
            }(std::make_index_sequence<sizeof...(Dims) - 1>{});
        } else {
            // For 1D tensors
            return this->template subview<Dims...>(slice{0, size()});
        }
    }

  private:
    template <std::size_t... Is, typename... Indices>
    static constexpr size_t calculate_index(std::index_sequence<Is...> /*unused*/, Indices... indices) {
        constexpr std::array<size_t, sizeof...(Dims)> dims = {Dims...};
        size_t index = 0;
        size_t stride = 1;
        ((index += (L == layout::row_major ? std::get<sizeof...(Dims) - 1 - Is>(std::forward_as_tuple(indices...))
                                           : std::get<Is>(std::forward_as_tuple(indices...))) *
                   stride,
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

    template <fixed_shape_tensor BlockTensor> constexpr void copy_from_block(const BlockTensor &block) {
        auto iter = detail::make_subviews_iterator(block, *this).begin();
        constexpr std::size_t num_repeats = total_size / BlockTensor::size();
        for (std::size_t i = 0; i < num_repeats; ++i) {
            *iter = block;
            ++iter;
        }
    }

    template <fixed_shape_tensor BlockTensor, std::size_t N>
    constexpr void copy_from_blocks(const std::array<BlockTensor, N> &blocks) {
        auto iter = detail::make_subviews_iterator(blocks[0], *this).begin();
        for (const auto &block : blocks) {
            *iter = block;
            ++iter;
        }
    }
    template <fixed_shape_tensor BlockTensor, std::size_t... Is>
    static constexpr bool check_dimensions(index_sequence<Is...> /*unused*/) {
        return (... && (constexpr_shape()[Is] % BlockTensor::constexpr_shape()[Is] == 0));
    }
};

// Vector types
template <typename T> using vec2_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 2>;
template <typename T> using vec3_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 3>;
template <typename T> using vec4_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 4>;
using ivec2 = vec2_t<int>;
using ivec3 = vec3_t<int>;
using ivec4 = vec4_t<int>;
using uvec2 = vec2_t<unsigned char>;
using uvec3 = vec3_t<unsigned char>;
using uvec4 = vec4_t<unsigned char>;
using vec2 = vec2_t<float>;
using vec3 = vec3_t<float>;
using vec4 = vec4_t<float>;
using dvec2 = vec2_t<double>;
using dvec3 = vec3_t<double>;
using dvec4 = vec4_t<double>;
using bvec2 = vec2_t<bool>;
using bvec3 = vec3_t<bool>;
using bvec4 = vec4_t<bool>;

// Square matrix types
template <typename T> using mat2_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 2>;
template <typename T> using mat3_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 3>;
template <typename T> using mat4_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 4>;
using imat2 = mat2_t<int>;
using imat3 = mat3_t<int>;
using imat4 = mat4_t<int>;
using umat2 = mat2_t<unsigned char>;
using umat3 = mat3_t<unsigned char>;
using umat4 = mat4_t<unsigned char>;
using mat2 = mat2_t<float>;
using mat3 = mat3_t<float>;
using mat4 = mat4_t<float>;
using dmat2 = mat2_t<double>;
using dmat3 = mat3_t<double>;
using dmat4 = mat4_t<double>;
using bmat2 = mat2_t<bool>;
using bmat3 = mat3_t<bool>;
using bmat4 = mat4_t<bool>;

// Non-square matrix types
template <typename T> using mat2x3_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 3>;
template <typename T> using mat2x4_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 4>;
template <typename T> using mat3x2_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 2>;
template <typename T> using mat3x4_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 4>;
template <typename T> using mat4x2_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 2>;
template <typename T> using mat4x3_t = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 3>;
using imat2x3 = mat2x3_t<int>;
using imat2x4 = mat2x4_t<int>;
using imat3x2 = mat3x2_t<int>;
using imat3x4 = mat3x4_t<int>;
using imat4x2 = mat4x2_t<int>;
using imat4x3 = mat4x3_t<int>;
using umat2x3 = mat2x3_t<unsigned char>;
using umat2x4 = mat2x4_t<unsigned char>;
using umat3x2 = mat3x2_t<unsigned char>;
using umat3x4 = mat3x4_t<unsigned char>;
using umat4x2 = mat4x2_t<unsigned char>;
using umat4x3 = mat4x3_t<unsigned char>;
using mat2x3 = mat2x3_t<float>;
using mat2x4 = mat2x4_t<float>;
using mat3x2 = mat3x2_t<float>;
using mat3x4 = mat3x4_t<float>;
using mat4x2 = mat4x2_t<float>;
using mat4x3 = mat4x3_t<float>;
using dmat2x3 = mat2x3_t<double>;
using dmat2x4 = mat2x4_t<double>;
using dmat3x2 = mat3x2_t<double>;
using dmat3x4 = mat3x4_t<double>;
using dmat4x2 = mat4x2_t<double>;
using dmat4x3 = mat4x3_t<double>;
using bmat2x3 = mat2x3_t<bool>;
using bmat2x4 = mat2x4_t<bool>;
using bmat3x2 = mat3x2_t<bool>;
using bmat3x4 = mat3x4_t<bool>;
using bmat4x2 = mat4x2_t<bool>;
using bmat4x3 = mat4x3_t<bool>;

// General tensor shapes
template <typename T, std::size_t... Dims>
using ndarr_t = fixed_tensor<T, layout::column_major, error_checking::disabled, Dims...>;
template <std::size_t... Dims> using indarr = ndarr_t<int, Dims...>;
template <std::size_t... Dims> using undarr = ndarr_t<unsigned char, Dims...>;
template <std::size_t... Dims> using ndarr = ndarr_t<float, Dims...>;
template <std::size_t... Dims> using dndarr = ndarr_t<double, Dims...>;
template <std::size_t... Dims> using bndarr = ndarr_t<bool, Dims...>;

} // namespace squint

#endif // SQUINT_FIXED_TENSOR_HPP