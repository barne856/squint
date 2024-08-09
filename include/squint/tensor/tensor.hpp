#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/tensor/iterable_tensor.hpp"
#include "squint/util/array_utils.hpp"
#include "squint/util/type_traits.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace squint {

// Concept for compile-time shapes
template <typename T>
concept compile_time_shape = is_index_sequence<T>::value;

// Concept for runtime shapes
template <typename T>
concept runtime_shape = std::is_same_v<T, std::vector<std::size_t>>;

// Unified Tensor class
// TODO
// - product for std::vector
// - how to handle GPU specialized tensor with overide subscript operator?
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled, bool OwnData = true>
class tensor : public iterable_tensor<tensor<T, Shape, Strides, ErrorChecking, OwnData>> {
  private:
    // Data storage
    std::conditional_t<
        OwnData, std::conditional_t<compile_time_shape<Shape>, std::array<T, product(Shape{})>, std::vector<T>>, T *>
        data_;

    // Shape and strides storage
    [[no_unique_address]] std::conditional_t<compile_time_shape<Shape>, std::integral_constant<Shape, Shape{}>, Shape>
        shape_;

    [[no_unique_address]] std::conditional_t<compile_time_shape<Strides>, std::integral_constant<Strides, Strides{}>,
                                             Strides> strides_;

  public:
    using value_type = T;
    using shape_type = Shape;
    using strides_type = Strides;
    using index_type =
        std::conditional_t<compile_time_shape<Shape>, std::array<std::size_t, Shape::size()>, std::vector<std::size_t>>;

    tensor()
        requires OwnData
    = default;

    // construct from initializer list
    tensor(std::initializer_list<T> init) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (init.size() != product(Shape{})) {
                throw std::invalid_argument("Initializer list size does not match tensor size");
            }
        }
        if constexpr (compile_time_shape<Shape>) {
            std::copy(init.begin(), init.end(), data_.begin());
        } else {
            data_ = std::vector<T>(init.begin(), init.end());
        }
    }

    // Constructor for runtime shapes
    tensor(Shape shape, Strides strides)
        requires(runtime_shape<Shape> && runtime_shape<Strides> && OwnData)
        : shape_(std::move(shape)), strides_(std::move(strides)) {
        data_.resize(product(shape_));
    }

    // Constructor for non-owned data (views)
    tensor(T *data, Shape shape, Strides strides)
        requires(!OwnData)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {}

    // Rank of the tensor
    [[nodiscard]] constexpr std::size_t rank() const {
        if constexpr (compile_time_shape<Shape>) {
            return Shape::size();
        } else {
            return shape_.size();
        }
    }

    // Shape of the tensor
    [[nodiscard]] constexpr auto shape() const {
        if constexpr (compile_time_shape<Shape>) {
            return make_array(Shape{});
        } else {
            return shape_;
        }
    }

    // Strides of the tensor
    [[nodiscard]] constexpr auto strides() const {
        if constexpr (compile_time_shape<Strides>) {
            return make_array(Strides{});
        } else {
            return strides_;
        }
    }

    // Total number of elements
    [[nodiscard]] constexpr std::size_t size() const {
        if constexpr (compile_time_shape<Shape>) {
            return product(Shape{});
        } else {
            return product(shape_);
        }
    }

    // Data access
    [[nodiscard]] constexpr T *data() {
        if constexpr (OwnData) {
            return data_.data();
        } else {
            return data_;
        }
    }

    [[nodiscard]] constexpr const T *data() const {
        if constexpr (OwnData) {
            return data_.data();
        } else {
            return data_;
        }
    }

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
        return data()[compute_offset(indices)];
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
        return data()[compute_offset(indices)];
    }

    // Error checking method
    static constexpr auto error_checking() -> error_checking { return ErrorChecking; }

  private:
    // Helper function to compute offset
    template <typename... Indices> [[nodiscard]] constexpr std::size_t compute_offset(const index_type &indices) const {
        std::size_t offset = 0;
        if constexpr (compile_time_shape<Shape>) {
            constexpr auto strides_array = make_array(Strides{});
            for (std::size_t i = 0; i < rank(); ++i) {
                offset += indices[i] * strides_array[i];
            }
        } else {
            for (std::size_t i = 0; i < rank(); ++i) {
                offset += indices[i] * strides_[i];
            }
        }
        return offset;
    }
};

// Type aliases for common use cases
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled>
using owned_tensor = tensor<T, Shape, Strides, ErrorChecking, true>;

template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled>
using tensor_view = tensor<T, Shape, Strides, ErrorChecking, false>;

// tens
template <typename T, std::size_t... Dims> using tens_t = tensor<T, std::index_sequence<Dims...>>;
template <std::size_t... Dims> using tens = tens_t<float, Dims...>;

} // namespace squint