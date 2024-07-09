#ifndef SQUINT_FIXED_TENSOR_HPP
#define SQUINT_FIXED_TENSOR_HPP

#include "squint/iterable_tensor.hpp"
#include "squint/quantity.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <array>
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
class fixed_tensor : public iterable_tensor<fixed_tensor<T, L, ErrorChecking, Dims...>, T, ErrorChecking> {
    static constexpr std::size_t total_size = (Dims * ...);
    std::array<T, total_size> data_;

  public:
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
        static_assert(check_dimensions<BlockTensor>(std::make_index_sequence<BlockTensor::constexpr_shape().size()>{}),
                      "Each dimension of the block must be a multiple of the corresponding dimension of the tensor");
        copy_from_block(block);
    }

    // Create a tensor from a list of tensor blocks or views
    template <fixed_shape_tensor BlockTensor, std::size_t N>
    constexpr fixed_tensor(const std::array<BlockTensor, N> &blocks) {
        static_assert(check_dimensions<BlockTensor>(std::make_index_sequence<BlockTensor::constexpr_shape().size()>{}),
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
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
    }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return data_[calculate_index(std::index_sequence_for<Indices...>{}, indices...)];
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
template <typename T> using vec2 = fixed_tensor<T, layout::column_major, error_checking::disabled, 2>;
template <typename T> using vec3 = fixed_tensor<T, layout::column_major, error_checking::disabled, 3>;
template <typename T> using vec4 = fixed_tensor<T, layout::column_major, error_checking::disabled, 4>;

// Square matrix types
template <typename T> using mat2 = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 2>;
template <typename T> using mat3 = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 3>;
template <typename T> using mat4 = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 4>;

// Non-square matrix types
template <typename T> using mat2x3 = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 3>;
template <typename T> using mat2x4 = fixed_tensor<T, layout::column_major, error_checking::disabled, 2, 4>;
template <typename T> using mat3x2 = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 2>;
template <typename T> using mat3x4 = fixed_tensor<T, layout::column_major, error_checking::disabled, 3, 4>;
template <typename T> using mat4x2 = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 2>;
template <typename T> using mat4x3 = fixed_tensor<T, layout::column_major, error_checking::disabled, 4, 3>;

} // namespace squint

#endif // SQUINT_FIXED_TENSOR_HPP