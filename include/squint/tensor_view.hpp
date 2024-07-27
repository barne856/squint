#ifndef SQUINT_TENSOR_VIEW_HPP
#define SQUINT_TENSOR_VIEW_HPP

#include "squint/iterable_tensor.hpp"
#include "squint/linear_algebra.hpp"
#include <array>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

// Compile-time utilities
template <std::size_t... Is> using index_sequence = std::index_sequence<Is...>;

template <layout L, std::size_t... Dims> struct compile_time_strides {
    static constexpr auto value = []() {
        std::array<std::size_t, sizeof...(Dims)> strides{};
        std::size_t stride = 1;
        if constexpr (L == layout::row_major) {
            for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= std::array<std::size_t, sizeof...(Dims)>{Dims...}[i];
            }
        } else {
            for (std::size_t i = 0; i < sizeof...(Dims); ++i) {
                strides[i] = stride;
                stride *= std::array<std::size_t, sizeof...(Dims)>{Dims...}[i];
            }
        }
        return strides;
    }();
};

template <layout L, typename Strides, std::size_t... Dims> struct compile_time_view_strides {
    static constexpr auto value = []() {
        constexpr std::size_t new_rank = sizeof...(Dims);
        std::array<std::size_t, new_rank> strides{};
        std::size_t j = 0;
        std::size_t i = 0;
        (((Dims > 0 ? (strides[j++] = Strides::value[i]) : 0), ++i), ...);
        return strides;
    }();
};

// Base class for all tensor views
template <typename Derived, typename T, error_checking ErrorChecking>
class tensor_view_base : public iterable_tensor<Derived, T, ErrorChecking> {
  protected:
    T *data_;

  public:
    constexpr tensor_view_base(T *data) : data_(data) {}

    constexpr T *data() { return data_; }
    constexpr const T *data() const { return data_; }

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
};

// Base class for fixed tensor views
template <typename Derived, typename T, layout L, typename Strides, error_checking ErrorChecking, std::size_t... Dims>
class fixed_tensor_view_base : public tensor_view_base<Derived, T, ErrorChecking>,
                               public fixed_linear_algebra_mixin<Derived, ErrorChecking> {
  protected:
    using tensor_view_base<Derived, T, ErrorChecking>::data_;

  public:
    using tensor_view_base<Derived, T, ErrorChecking>::tensor_view_base;
    template <tensor OtherTensor> auto &operator=(const OtherTensor &other) {
        constexpr auto this_shape = std::array<std::size_t, sizeof...(Dims)>({Dims...});
        constexpr auto other_shape = OtherTensor::constexpr_shape();
        constexpr auto min_rank = std::min(sizeof...(Dims), other_shape.size());

        // Compile-time check for matching dimensions
        static_assert(
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ((Is >= min_rank || this_shape[Is] == other_shape[Is]) && ...);
            }(std::make_index_sequence<sizeof...(Dims)>{}),
            "Incompatible shape for assignment: dimensions must match where they overlap");

        auto this_it = static_cast<Derived *>(this)->begin();
        auto other_it = other.begin();
        for (; this_it != static_cast<Derived *>(this)->end(); ++this_it, ++other_it) {
            *this_it = *other_it;
        }
        return *static_cast<Derived *>(this);
    }

    static constexpr std::size_t rank() { return sizeof...(Dims); }
    static constexpr std::size_t size() { return (Dims * ...); }
    static constexpr auto constexpr_shape() { return std::array<std::size_t, sizeof...(Dims)>{Dims...}; }
    static constexpr std::vector<std::size_t> shape() { return {Dims...}; }
    static constexpr std::vector<std::size_t> strides() {
        auto strides_array = Strides::value;
        return std::vector<std::size_t>(std::begin(strides_array), std::end(strides_array));
    }
    static constexpr std::array<std::size_t, sizeof...(Dims)> constexpr_strides() { return Strides::value; }
    static constexpr layout get_layout() { return L; }
    static constexpr error_checking get_error_checking() { return ErrorChecking; }

    template <typename... Indices> constexpr const T &at(Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return data_[calculate_offset(std::index_sequence_for<Indices...>{}, indices...)];
    }

    const T &at_impl(const std::vector<std::size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(indices);
        }
        return data_[calculate_offset_runtime(indices)];
    }

    template <std::size_t... NewDims, typename... Offset> constexpr auto subview(Offset... start) const {
        static_assert(sizeof...(NewDims) == sizeof...(Offset), "Number of new dimensions must match number of offsets");
        static_assert(sizeof...(NewDims) <= sizeof...(Dims), "Too many offset arguments");
        if constexpr (ErrorChecking == error_checking::enabled) {
            std::vector<std::size_t> new_shape = {NewDims...}; 
            std::vector<std::size_t> new_indices = {static_cast<std::size_t>(start)...};
            this->check_subview_bounds(new_shape, new_indices);
        }
        return create_subview<NewDims...>(std::make_index_sequence<sizeof...(NewDims)>{}, start...);
    }

    constexpr auto view() const { return *this; }

    template <std::size_t... NewDims> auto reshape() const {
        static_assert((NewDims * ...) == size(), "New shape must have the same total size");
        using new_strides = compile_time_strides<L, NewDims...>;
        return const_fixed_tensor_view<T, L, new_strides, ErrorChecking, NewDims...>(data_);
    }

    auto flatten() const { return const_fixed_tensor_view<T, L, Strides, ErrorChecking, size()>(data_); }

  protected:
    template <std::size_t... Is, typename... Indices>
    static constexpr std::size_t calculate_offset(std::index_sequence<Is...> /*unused*/, Indices... indices) {
        return ((indices * Strides::value[Is]) + ... + 0);
    }

    std::size_t calculate_offset_runtime(const std::vector<std::size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(indices);
        }

        std::size_t offset = 0;
        for (std::size_t i = 0; i < sizeof...(Dims); ++i) {
            offset += indices[i] * Strides::value[i];
        }
        return offset;
    }

  private:
    template <std::size_t... NewDims, std::size_t... Is, typename... Offset>
    constexpr auto create_subview(std::index_sequence<Is...> /*unused*/, Offset... start) const {
        auto start_arr = std::array<std::size_t, sizeof...(Offset)>{static_cast<std::size_t>(start)...};
        T *new_data = data_ + (0 + ... + (start_arr[Is] * Strides::value[Is]));

        using new_strides = compile_time_view_strides<L, Strides, NewDims...>;

        return const_fixed_tensor_view<T, L, new_strides, ErrorChecking, NewDims...>(new_data);
    }
};

// Fixed tensor view
template <typename T, layout L, typename Strides, error_checking ErrorChecking, std::size_t... Dims>
class fixed_tensor_view : public fixed_tensor_view_base<fixed_tensor_view<T, L, Strides, ErrorChecking, Dims...>, T, L,
                                                        Strides, ErrorChecking, Dims...> {
    using base_type = fixed_tensor_view_base<fixed_tensor_view<T, L, Strides, ErrorChecking, Dims...>, T, L, Strides,
                                             ErrorChecking, Dims...>;

  public:
    using base_type::operator=;
    using base_type::base_type;
    using base_type::get_layout;
    using base_type::rank;
    using base_type::shape;
    using base_type::size;
    using base_type::strides;
    using base_type::subview;

    // assignment operator
    auto &operator=(const fixed_tensor_view &other) { return base_type::operator=(other); }

    // Non-const version of at
    template <typename... Indices> constexpr T &at(Indices... indices) {
        return const_cast<T &>(base_type::at(indices...));
    }

    using base_type::at;
    using base_type::at_impl;

    T &at_impl(const std::vector<std::size_t> &indices) { return const_cast<T &>(base_type::at_impl(indices)); }

    template <std::size_t... NewDims, typename... Offset> constexpr auto subview(Offset... start) {
        static_assert(sizeof...(NewDims) == sizeof...(Offset), "Number of new dimensions must match number of offsets");
        static_assert(sizeof...(NewDims) <= sizeof...(Dims), "Too many offset arguments");
        if constexpr (ErrorChecking == error_checking::enabled) {
            std::vector<std::size_t> new_shape = {NewDims...}; 
            std::vector<std::size_t> new_indices = {static_cast<std::size_t>(start)...};
            this->check_subview_bounds(new_shape, new_indices);
        }
        return create_subview<NewDims...>(std::make_index_sequence<sizeof...(NewDims)>{}, start...);
    }

    constexpr auto view() const { return *this; }

    template <std::size_t... NewDims> auto reshape() {
        static_assert((NewDims * ...) == size(), "New shape must have the same total size");
        using new_strides = compile_time_strides<L, NewDims...>;
        return fixed_tensor_view<T, L, new_strides, ErrorChecking, NewDims...>(data_);
    }

    auto flatten() { return fixed_tensor_view<T, L, Strides, ErrorChecking, size()>(data_); }

  private:
    using base_type::data_;
    template <std::size_t... NewDims, std::size_t... Is, typename... Offset>
    constexpr auto create_subview(std::index_sequence<Is...> /*unused*/, Offset... start) {
        auto start_arr = std::array<std::size_t, sizeof...(Offset)>{static_cast<std::size_t>(start)...};
        T *new_data = data_ + (0 + ... + (start_arr[Is] * Strides::value[Is]));

        using new_strides = compile_time_view_strides<L, Strides, NewDims...>;

        return fixed_tensor_view<T, L, new_strides, ErrorChecking, NewDims...>(new_data);
    }
};

// Const fixed tensor view
template <typename T, layout L, typename Strides, error_checking ErrorChecking, std::size_t... Dims>
class const_fixed_tensor_view
    : public fixed_tensor_view_base<const_fixed_tensor_view<T, L, Strides, ErrorChecking, Dims...>, const T, L, Strides,
                                    ErrorChecking, Dims...> {
    using base_type = fixed_tensor_view_base<const_fixed_tensor_view<T, L, Strides, ErrorChecking, Dims...>, const T, L,
                                             Strides, ErrorChecking, Dims...>;

  public:
    using base_type::at;
    using base_type::at_impl;
    using base_type::base_type;
    using base_type::get_layout;
    using base_type::rank;
    using base_type::shape;
    using base_type::size;
    using base_type::strides;
    using base_type::subview;
};

// Base class for dynamic tensor views
template <typename Derived, typename T, error_checking ErrorChecking>
class dynamic_tensor_view_base : public tensor_view_base<Derived, T, ErrorChecking>,
                                 public dynamic_linear_algebra_mixin<Derived, ErrorChecking> {
  protected:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    layout layout_;

    using tensor_view_base<Derived, T, ErrorChecking>::data_;

  public:
    using tensor_view_base<Derived, T, ErrorChecking>::tensor_view_base;
    template <tensor OtherTensor> auto &operator=(const OtherTensor &other) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            // dimensions must match where they overlap
            auto this_shape = shape();
            auto other_shape = other.shape();
            auto min_rank = std::min(this_shape.size(), other_shape.size());

            if (!std::equal(this_shape.begin(), this_shape.begin() + min_rank, other_shape.begin())) {
                throw std::invalid_argument(
                    "Incompatible shape for assignment: dimensions must match where they overlap");
            }
        }

        auto this_it = static_cast<Derived *>(this)->begin();
        auto other_it = other.begin();
        for (; this_it != static_cast<Derived *>(this)->end(); ++this_it, ++other_it) {
            *this_it = *other_it;
        }
        return *static_cast<Derived *>(this);
    }
    dynamic_tensor_view_base(T *data, std::vector<std::size_t> shape, std::vector<std::size_t> strides, layout l)
        : tensor_view_base<Derived, T, ErrorChecking>(data), shape_(std::move(shape)), strides_(std::move(strides)),
          layout_(l) {}

    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()); }
    std::vector<std::size_t> shape() const { return shape_; }
    std::vector<std::size_t> strides() const { return strides_; }
    layout get_layout() const { return layout_; }
    static constexpr error_checking get_error_checking() { return ErrorChecking; }

    template <typename... Indices> const T &at(Indices... indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(std::vector<size_t>{static_cast<size_t>(indices)...});
        }
        return data_[calculate_offset(std::vector<std::size_t>{static_cast<std::size_t>(indices)...})];
    }

    const T &at_impl(const std::vector<std::size_t> &indices) const { return data_[calculate_offset(indices)]; }

    auto subview(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& start) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds(shape, start);
        }
        return create_subview(shape, start);
    }

    constexpr auto view() const { return *this; }

    auto flatten() const {
        return const_dynamic_tensor_view<T, ErrorChecking>(data_, std::vector<std::size_t>{size()},
                                                           std::vector<std::size_t>{1}, layout_);
    }

  protected:
    std::size_t calculate_offset(const std::vector<std::size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_bounds(indices);
        }

        std::size_t offset = 0;
        for (std::size_t i = 0; i < rank(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    void process_slice(std::size_t size, std::size_t start, std::vector<std::size_t> &new_shape, std::vector<std::size_t> &new_strides,
                       T *&new_data, std::size_t &i) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (i >= rank()) {
                throw std::out_of_range("Too many slice arguments");
            }
        }
            new_shape.push_back(size);
            new_strides.push_back(strides_[i]);
            new_data += start * strides_[i];
        ++i;
    }

  private:
    auto create_subview(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& start) const {
        std::vector<std::size_t> new_shape;
        std::vector<std::size_t> new_strides;
        T *new_data = data_;

        std::size_t i = 0;
        for (std::size_t j = 0; j < shape.size(); ++j) {
            this->process_slice(shape[j],start[j], new_shape, new_strides, new_data, i);
        }

        return const_dynamic_tensor_view<T, ErrorChecking>(new_data, std::move(new_shape), std::move(new_strides),
                                                           layout_);
    }
};

// Dynamic tensor view
template <typename T, error_checking ErrorChecking>
class dynamic_tensor_view : public dynamic_tensor_view_base<dynamic_tensor_view<T, ErrorChecking>, T, ErrorChecking> {
    using base_type = dynamic_tensor_view_base<dynamic_tensor_view<T, ErrorChecking>, T, ErrorChecking>;

  public:
    using base_type::operator=;
    using base_type::base_type;
    using base_type::get_layout;
    using base_type::rank;
    using base_type::shape;
    using base_type::size;
    using base_type::strides;
    using base_type::subview;

    // assignment operator
    auto &operator=(const dynamic_tensor_view &other) { return base_type::operator=(other); }

    template <typename... Indices> T &at(Indices... indices) { return const_cast<T &>(base_type::at(indices...)); }

    using base_type::at;
    using base_type::at_impl;

    T &at_impl(const std::vector<std::size_t> &indices) { return const_cast<T &>(base_type::at_impl(indices)); }

    // Non-const version of subview
    auto subview(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& start) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds(shape, start);
        }
        return create_subview(shape, start);
    }

    constexpr auto view() { return *this; }

    void reshape(std::vector<std::size_t> new_shape) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>()) != base_type::size()) {
                throw std::invalid_argument("New shape must have the same total size");
            }
        }
        base_type::shape_ = std::move(new_shape);
    }

    auto flatten() {
        return dynamic_tensor_view<T, ErrorChecking>(base_type::data_, std::vector<std::size_t>{size()},
                                                     std::vector<std::size_t>{1}, base_type::layout_);
    }

  private:
    auto create_subview(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& start) {
        std::vector<std::size_t> new_shape;
        std::vector<std::size_t> new_strides;
        T *new_data = base_type::data_;

        std::size_t i = 0;
        for (std::size_t j = 0; j < shape.size(); ++j) {
            this->process_slice(shape[j], start[j], new_shape, new_strides, new_data, i);
        }

        return dynamic_tensor_view<T, ErrorChecking>(new_data, std::move(new_shape), std::move(new_strides),
                                                     base_type::layout_);
    }
};

// Const dynamic tensor view
template <typename T, error_checking ErrorChecking>
class const_dynamic_tensor_view
    : public dynamic_tensor_view_base<const_dynamic_tensor_view<T, ErrorChecking>, const T, ErrorChecking> {
    using base_type = dynamic_tensor_view_base<const_dynamic_tensor_view<T, ErrorChecking>, const T, ErrorChecking>;

  public:
    using base_type::at;
    using base_type::at_impl;
    using base_type::base_type;
    using base_type::get_layout;
    using base_type::rank;
    using base_type::shape;
    using base_type::size;
    using base_type::strides;
    using base_type::subview;
};

// Helper functions to create fixed tensor views
template <typename T, layout L, error_checking ErrorChecking, std::size_t... Dims>
constexpr auto make_fixed_tensor_view(fixed_tensor<T, L, ErrorChecking, Dims...> &tensor) {
    using initial_strides = compile_time_strides<L, Dims...>;
    return fixed_tensor_view<T, L, initial_strides, ErrorChecking, Dims...>(tensor.data());
}

template <typename T, layout L, error_checking ErrorChecking, std::size_t... Dims>
constexpr auto make_fixed_tensor_view(const fixed_tensor<T, L, ErrorChecking, Dims...> &tensor) {
    using initial_strides = compile_time_strides<L, Dims...>;
    return const_fixed_tensor_view<T, L, initial_strides, ErrorChecking, Dims...>(tensor.data());
}

// Helper functions to create dynamic tensor views
template <typename T, error_checking ErrorChecking>
auto make_dynamic_tensor_view(dynamic_tensor<T, ErrorChecking> &tensor) {
    return dynamic_tensor_view<T, ErrorChecking>(tensor.data(), tensor.shape(), tensor.strides(), tensor.get_layout());
}

template <typename T, error_checking ErrorChecking>
auto make_dynamic_tensor_view(const dynamic_tensor<T, ErrorChecking> &tensor) {
    return const_dynamic_tensor_view<T, ErrorChecking>(tensor.data(), tensor.shape(), tensor.strides(),
                                                       tensor.get_layout());
}

} // namespace squint

#endif // SQUINT_TENSOR_VIEW_HPP