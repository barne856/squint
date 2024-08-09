#include "squint/core/error_checking.hpp"
#include "squint/util/array_utils.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace squint {

// Concepts for compile-time and runtime shapes
template <typename T>
concept compile_time_shape = is_index_sequence<T>::value;

template <typename T>
concept runtime_shape = std::is_same_v<T, std::vector<std::size_t>>;

// Base tensor class using CRTP
template <typename Derived, typename T, typename Shape, typename Strides, error_checking ErrorChecking>
class base_tensor {
  protected:
    using value_type = T;
    using shape_type = Shape;
    using strides_type = Strides;
    using index_type =
        std::conditional_t<compile_time_shape<Shape>, std::array<std::size_t, Shape::size()>, std::vector<std::size_t>>;

  public:
    [[nodiscard]] constexpr std::size_t rank() const { return static_cast<const Derived *>(this)->rank_impl(); }

    [[nodiscard]] constexpr auto shape() const { return static_cast<const Derived *>(this)->shape_impl(); }

    [[nodiscard]] constexpr auto strides() const { return static_cast<const Derived *>(this)->strides_impl(); }

    [[nodiscard]] constexpr std::size_t size() const { return static_cast<const Derived *>(this)->size_impl(); }

    [[nodiscard]] constexpr T *data() { return static_cast<Derived *>(this)->data_impl(); }

    [[nodiscard]] constexpr const T *data() const { return static_cast<const Derived *>(this)->data_impl(); }

    static constexpr error_checking error_checking() { return ErrorChecking; }

    constexpr T &operator[](const index_type &indices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            static_cast<Derived *>(this)->check_indices(indices);
        }
        return this->data()[this->compute_offset(indices)];
    }

    constexpr const T &operator[](const index_type &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            static_cast<Derived *>(this)->check_indices(indices);
        }
        return this->data()[this->compute_offset(indices)];
    }

  protected:
    // Helper function to compute offset
    template <typename... Indices> [[nodiscard]] constexpr std::size_t compute_offset(Indices... indices) const {
        return static_cast<const Derived *>(this)->compute_offset_impl(indices...);
    }

    // Helper function to check indices (only called when ErrorChecking is enabled)
    template <typename... Indices> constexpr void check_indices(Indices... indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            static_cast<const Derived *>(this)->check_indices_impl(indices...);
        }
    }
};

// Fixed tensor class
template <typename Derived, typename T, typename Shape, typename Strides, error_checking ErrorChecking>
class fixed_tensor : public base_tensor<Derived, T, Shape, Strides, ErrorChecking> {
  protected:
    std::array<T, product(Shape{})> data_;
    static constexpr auto shape_ = make_array(Shape{});
    static constexpr auto strides_ = make_array(Strides{});

  public:
    using typename base_tensor<Derived, T, Shape, Strides, ErrorChecking>::index_type;

    [[nodiscard]] constexpr std::size_t rank_impl() const { return Shape::size(); }

    [[nodiscard]] constexpr auto shape_impl() const { return shape_; }

    [[nodiscard]] constexpr auto strides_impl() const { return strides_; }

    [[nodiscard]] constexpr std::size_t size_impl() const { return product(Shape{}); }

    [[nodiscard]] constexpr T *data_impl() { return data_.data(); }

    [[nodiscard]] constexpr const T *data_impl() const { return data_.data(); }

  protected:
    template <typename... Indices> [[nodiscard]] constexpr std::size_t compute_offset_impl(Indices... indices) const {
        return (... + (indices * strides_[sizeof...(Indices) - 1 - __builtin_ctz((unsigned)sizeof...(Indices))]));
    }

    template <typename... Indices> constexpr void check_indices_impl(Indices... indices) const {
        (check_index_impl(indices, shape_[sizeof...(Indices) - 1 - __builtin_ctz((unsigned)sizeof...(Indices))]),
         ...);
    }

    constexpr void check_index_impl(std::size_t index, std::size_t dim_size) const {
        if (index >= dim_size) {
            throw std::out_of_range("Index out of bounds");
        }
    }
};

// Dynamic tensor class
template <typename Derived, typename T, error_checking ErrorChecking>
class dynamic_tensor
    : public base_tensor<Derived, T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking> {
  protected:
    std::vector<T> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

  public:
    using
        typename base_tensor<Derived, T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking>::index_type;

    [[nodiscard]] std::size_t rank_impl() const { return shape_.size(); }

    [[nodiscard]] const auto &shape_impl() const { return shape_; }

    [[nodiscard]] const auto &strides_impl() const { return strides_; }

    [[nodiscard]] std::size_t size_impl() const { return data_.size(); }

    [[nodiscard]] T *data_impl() { return data_.data(); }

    [[nodiscard]] const T *data_impl() const { return data_.data(); }

  protected:
    template <typename... Indices> [[nodiscard]] std::size_t compute_offset_impl(Indices... indices) const {
        return (... + (indices * strides_[sizeof...(Indices) - 1 - __builtin_ctz((unsigned)sizeof...(Indices))]));
    }

    template <typename... Indices> void check_indices_impl(Indices... indices) const {
        std::size_t i = 0;
        (check_index_impl(indices, shape_[i++]), ...);
    }

    void check_index_impl(std::size_t index, std::size_t dim_size) const {
        if (index >= dim_size) {
            throw std::out_of_range("Index out of bounds");
        }
    }
};

// Owned fixed tensor
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled>
class owned_fixed_tensor
    : public fixed_tensor<owned_fixed_tensor<T, Shape, Strides, ErrorChecking>, T, Shape, Strides, ErrorChecking> {
  public:
    owned_fixed_tensor() = default;

    owned_fixed_tensor(std::initializer_list<T> init) {
        if (init.size() != this->size()) {
            throw std::invalid_argument("Initializer list size does not match tensor size");
        }
        std::copy(init.begin(), init.end(), this->data_.begin());
    }
};

// View fixed tensor
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled>
class view_fixed_tensor
    : public fixed_tensor<view_fixed_tensor<T, Shape, Strides, ErrorChecking>, T, Shape, Strides, ErrorChecking> {
  private:
    T *data_ptr_;

  public:
    explicit view_fixed_tensor(T *data) : data_ptr_(data) {}

    [[nodiscard]] T *data_impl() { return data_ptr_; }

    [[nodiscard]] const T *data_impl() const { return data_ptr_; }
};

// Owned dynamic tensor
template <typename T, error_checking ErrorChecking = error_checking::disabled>
class owned_dynamic_tensor : public dynamic_tensor<owned_dynamic_tensor<T, ErrorChecking>, T, ErrorChecking> {
  public:
    owned_dynamic_tensor(std::vector<std::size_t> shape, std::vector<std::size_t> strides)
        : dynamic_tensor<owned_dynamic_tensor<T, ErrorChecking>, T, ErrorChecking>() {
        this->shape_ = std::move(shape);
        this->strides_ = std::move(strides);
        this->data_.resize(product(this->shape_));
    }
};

// View dynamic tensor
template <typename T, error_checking ErrorChecking = error_checking::disabled>
class view_dynamic_tensor : public dynamic_tensor<view_dynamic_tensor<T, ErrorChecking>, T, ErrorChecking> {
  private:
    T *data_ptr_;

  public:
    view_dynamic_tensor(T *data, std::vector<std::size_t> shape, std::vector<std::size_t> strides)
        : dynamic_tensor<view_dynamic_tensor<T, ErrorChecking>, T, ErrorChecking>(), data_ptr_(data) {
        this->shape_ = std::move(shape);
        this->strides_ = std::move(strides);
    }

    [[nodiscard]] T *data_impl() { return data_ptr_; }

    [[nodiscard]] const T *data_impl() const { return data_ptr_; }
};

// GPU tensor classes (outline)
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled>
class gpu_fixed_tensor;

template <typename T, error_checking ErrorChecking = error_checking::disabled> class gpu_dynamic_tensor;

// Type aliases for common use cases
template <typename T, std::size_t... Dims> using tens_t = owned_fixed_tensor<T, std::index_sequence<Dims...>>;
template <std::size_t... Dims> using tens = tens_t<float, Dims...>;

} // namespace squint