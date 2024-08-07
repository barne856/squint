#ifndef SQUINT_TENSOR_TENSOR_HPP
#define SQUINT_TENSOR_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/tensor/iterable_tensor.hpp"
#include "squint/util/array_utils.hpp"
#include "squint/util/type_traits.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace squint {

// Helper to compute the product of a range of values in a parameter pack
template <std::size_t Begin, std::size_t End, std::size_t... Dims> struct product_range {
    static constexpr std::size_t value = 1;
};

template <std::size_t Begin, std::size_t End, std::size_t First, std::size_t... Rest>
struct product_range<Begin, End, First, Rest...> {
    static constexpr std::size_t value = (Begin < End ? First : 1) * product_range<Begin + 1, End, Rest...>::value;
};

// Helper to compute a single stride
template <std::size_t Idx, layout Layout, std::size_t... Dims> struct compute_single_stride {
    static constexpr std::size_t value = Layout == layout::row_major
                                             ? product_range<Idx + 1, sizeof...(Dims), Dims...>::value
                                             : product_range<0, Idx, Dims...>::value;
};

// Helper to compute all strides
template <layout Layout, typename Seq, typename Shape> struct compute_strides;

template <layout Layout, std::size_t... Is, std::size_t... Dims>
struct compute_strides<Layout, std::index_sequence<Is...>, std::index_sequence<Dims...>> {
    using type = std::index_sequence<compute_single_stride<Is, Layout, Dims...>::value...>;
};

// TODO, make tensor_base without initalizers, data array, or subscript operators
// make dense_tensor with initalizers, data array, and subscript operators
// make tensor_view with initalizers, data ptr, and subscript operators
// make gpu_tensor with initalizers, data ptr
template <typename T, typename Shape, layout Layout = layout::column_major,
          error_checking ErrorChecking = error_checking::disabled>
class tensor : public iterable_tensor<tensor<T, Shape, Layout, ErrorChecking>> {
  public:
    using value_type = T;
    using shape_type = Shape;
    using index_type = std::conditional_t<is_index_sequence<Shape>::value, std::array<std::size_t, Shape::size()>,
                                          std::vector<std::size_t>>;
    // compute strides type as index sequence at compile time
    using strides_type = typename compute_strides<Layout, std::make_index_sequence<Shape::size()>, Shape>::type;

    tensor() = default;

    // construct from initializer list
    tensor(std::initializer_list<T> init) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (init.size() != product(Shape{})) {
                throw std::invalid_argument("Initializer list size does not match tensor size");
            }
        }
        if constexpr (is_index_sequence<Shape>::value) {
            std::copy(init.begin(), init.end(), data_.begin());
        } else {
            data_ = std::vector<T>(init.begin(), init.end());
        }
    }

    [[nodiscard]] constexpr auto rank() const -> std::size_t {
        if constexpr (is_index_sequence<Shape>::value) {
            return Shape::size();
        }
    }

    [[nodiscard]] constexpr auto size() const -> std::size_t {
        if constexpr (is_index_sequence<Shape>::value) {
            return product(Shape{});
        }
    }

    constexpr auto shape() const -> auto {
        if constexpr (is_index_sequence<Shape>::value) {
            return make_array(Shape{});
        }
    }

    constexpr auto strides() const -> auto {
        if constexpr (is_index_sequence<Shape>::value) {
            return make_array(strides_type{});
        }
    }

    auto data() const -> const T * { return data_.data(); }
    auto data() -> T * { return data_.data(); }

    static constexpr auto layout() -> layout { return Layout; }
    static constexpr auto error_checking() -> error_checking { return ErrorChecking; }

    auto operator[](const index_type &indices) -> T & {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (indices.size() != rank()) {
                throw std::out_of_range("Index size does not match tensor rank");
            }
            for (std::size_t i = 0; i < rank(); ++i) {
                if (indices[i] >= shape()[i]) {
                    throw std::out_of_range("Index out of range");
                }
            }
        }
        return data_[compute_offset(indices)];
    }

    auto operator[](const index_type &indices) const -> const T & {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (indices.size() != rank()) {
                throw std::out_of_range("Index size does not match tensor rank");
            }
            for (std::size_t i = 0; i < rank(); ++i) {
                if (indices[i] >= shape()[i]) {
                    throw std::out_of_range("Index out of range");
                }
            }
        }
        return data_[compute_offset(indices)];
    }

  private:
    using data_type =
        std::conditional_t<is_index_sequence<Shape>::value, std::array<T, product(Shape{})>, std::vector<T>>;
    data_type data_;

    constexpr auto compute_offset(const index_type &indices) const -> std::size_t {
        if constexpr (is_index_sequence<Shape>::value) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < rank(); ++i) {
                offset += indices[i] * strides()[i];
            }
            return offset;
        }
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_HPP