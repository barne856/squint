/**
 * @file tensor.hpp
 * @brief Defines the tensor class for multi-dimensional array operations.
 *
 * This file contains the implementation of the tensor class, which provides
 * a flexible and efficient representation of multi-dimensional arrays. The
 * tensor class supports both fixed and dynamic shapes, various memory layouts,
 * and different ownership models. It also includes functionality for creating
 * subviews and iterating over tensor elements.
 *
 * Key features:
 * - Support for fixed and dynamic tensor shapes
 * - Configurable error checking
 * - Flexible memory ownership (owner or reference)
 * - Support for different memory spaces (e.g., host, device)
 * - Subview creation and iteration
 *
 */

#ifndef SQUINT_TENSOR_HPP
#define SQUINT_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/iterable_tensor.hpp"
#include "squint/util/array_utils.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace squint {

/**
 * @brief A multi-dimensional tensor class with flexible shape, strides, and memory management.
 *
 * @tparam T The data type of the tensor elements.
 * @tparam Shape The shape type of the tensor, can be fixed or dynamic.
 * @tparam Strides The strides type of the tensor, defaults to column-major strides.
 * @tparam ErrorChecking Enum to enable or disable bounds checking.
 * @tparam OwnershipType Enum to specify whether the tensor owns its data or is a view.
 * @tparam MemorySpace Enum to specify the memory space of the tensor data.
 */
template <typename T, typename Shape, typename Strides = column_major_strides<Shape>,
          error_checking ErrorChecking = error_checking::disabled, ownership_type OwnershipType = ownership_type::owner,
          memory_space MemorySpace = memory_space::host>
class tensor : public iterable_mixin<tensor<T, Shape, Strides, ErrorChecking, OwnershipType, MemorySpace>> {
  private:
    /// @brief Type alias for shape storage, using std::integral_constant for fixed shapes.
    using shape_storage = std::conditional_t<fixed_shape<Shape>, std::integral_constant<Shape, Shape{}>, Shape>;
    /// @brief Type alias for strides storage, using std::integral_constant for fixed strides.
    using strides_storage =
        std::conditional_t<fixed_shape<Strides>, std::integral_constant<Strides, Strides{}>, Strides>;
    /// @brief Type alias for data storage, using std::array for fixed shapes and std::vector for dynamic shapes.
    using data_storage =
        std::conditional_t<OwnershipType == ownership_type::owner,
                           std::conditional_t<fixed_shape<Shape>, std::array<T, product(Shape{})>, std::vector<T>>,
                           T *>;

    // Shape and strides storage
    // These are effectively empty if the shape is fixed. [[no_unique_address]] is used to avoid increasing the size of
    // the tensor class. Note: the order here matters, MSVC does not appear to be able to optimize the size of the
    // tensor if these definitions come last in the classes member list.
    NO_UNIQUE_ADDRESS shape_storage shape_;     ///< Storage for tensor shape.
    NO_UNIQUE_ADDRESS strides_storage strides_; ///< Storage for tensor strides.
    data_storage data_;                         ///< Storage for tensor data.

  public:
    using value_type = T;         ///< The type of the tensor elements.
    using shape_type = Shape;     ///< The type used to represent the tensor shape.
    using strides_type = Strides; ///< The type used to represent the tensor strides.
    /// @brief The type used for indexing, std::array for fixed shapes, std::vector for dynamic shapes.
    using index_type =
        std::conditional_t<fixed_shape<Shape>, std::array<std::size_t, Shape::size()>, std::vector<std::size_t>>;

    /**
     * @brief Default constructor (only available for owner tensors).
     */
    tensor()
        requires(OwnershipType == ownership_type::owner)
    = default;

    /**
     * @brief Constructor from initializer list (only available for owner tensors).
     * @param init Initializer list containing tensor data.
     * @throws std::invalid_argument If the initializer list size doesn't match the tensor size (when ErrorChecking is
     * enabled).
     */
    tensor(std::initializer_list<T> init)
        requires(OwnershipType == ownership_type::owner)
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (init.size() != product(Shape{})) {
                throw std::invalid_argument("Initializer list size does not match tensor size");
            }
        }
        if constexpr (fixed_shape<Shape>) {
            std::copy(init.begin(), init.end(), data_.begin());
        } else {
            data_ = std::vector<T>(init.begin(), init.end());
        }
    }

    /**
     * @brief Constructs a tensor filled with a single value.
     *
     * @param value The value to fill the tensor with.
     */
    explicit tensor(const T &value)
        requires(OwnershipType == ownership_type::owner)
        : data_() {
        std::fill(data_.begin(), data_.end(), value);
    }

    /**
     * @brief Constructs a fixed shape tensor from a flat array of elements.
     *
     * @param elements A std::array containing the elements to initialize the tensor with.
     * @throws std::invalid_argument If the size of the input array doesn't match the tensor size (when ErrorChecking is
     * enabled).
     */
    tensor(const std::array<T, product(Shape{})> &elements)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner)
        : data_(elements) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (elements.size() != product(Shape{})) {
                throw std::invalid_argument("Input array size does not match tensor size");
            }
        }
    }

    /**
     * @brief Constructor for dynamic shape and strides (only available for owner tensors).
     * @param shape The shape of the tensor.
     * @param strides The strides of the tensor.
     */
    tensor(Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && dynamic_shape<Strides> && OwnershipType == ownership_type::owner)
        : shape_(std::move(shape)), strides_(std::move(strides)) {
        data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
    }

    /**
     * @brief Constructor for dynamic shape with specified layout (only available for owner tensors).
     * @param shape The shape of the tensor.
     * @param l The memory layout of the tensor (row-major or column-major).
     */
    tensor(Shape shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && dynamic_shape<Strides> && OwnershipType == ownership_type::owner)
        : shape_(std::move(shape)), strides_(compute_strides(l)) {
        data_.resize(std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>()));
    }

    /**
     * @brief Constructs a dynamic shape tensor from a vector of elements.
     *
     * @param shape A vector specifying the dimensions of the tensor.
     * @param elements A vector containing the elements to initialize the tensor with.
     * @param l The memory layout of the tensor (default is column_major).
     * @throws std::invalid_argument If the size of the input vector doesn't match the tensor size (when ErrorChecking
     * is enabled).
     */
    tensor(std::vector<size_t> shape, const std::vector<T> &elements, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner)
        : shape_(std::move(shape)), strides_(compute_strides(l)), data_(elements) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
            if (elements.size() != total_size) {
                throw std::invalid_argument("Input vector size does not match tensor size");
            }
        }
    }

    /**
     * @brief Constructor for reference tensors with dynamic shape and strides.
     * @param data Pointer to the tensor data.
     * @param shape The shape of the tensor.
     * @param strides The strides of the tensor.
     */
    tensor(T *data, Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && dynamic_shape<Strides> && OwnershipType == ownership_type::reference)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {}

    /**
     * @brief Constructor for reference tensors with fixed shape and strides.
     * @param data Pointer to the tensor data.
     */
    tensor(T *data)
        requires(fixed_shape<Shape> && fixed_shape<Strides> && OwnershipType == ownership_type::reference)
        : data_(data) {}

    /**
     * @brief Get the rank (number of dimensions) of the tensor.
     * @return The rank of the tensor.
     */
    [[nodiscard]] constexpr auto rank() const {
        if constexpr (fixed_shape<Shape>) {
            return Shape::size();
        } else {
            return shape_.size();
        }
    }

    /**
     * @brief Get the shape of the tensor.
     * @return An array or vector representing the tensor shape.
     */
    [[nodiscard]] constexpr auto shape() const {
        if constexpr (fixed_shape<Shape>) {
            return make_array(Shape{});
        } else {
            return shape_;
        }
    }

    /**
     * @brief Get the strides of the tensor.
     * @return An array or vector representing the tensor strides.
     */
    [[nodiscard]] constexpr auto strides() const {
        if constexpr (fixed_shape<Strides>) {
            return make_array(Strides{});
        } else {
            return strides_;
        }
    }

    /**
     * @brief Get the total number of elements in the tensor.
     * @return The total number of elements.
     */
    [[nodiscard]] constexpr auto size() const {
        if constexpr (fixed_shape<Shape>) {
            return product(Shape{});
        } else if constexpr (OwnershipType == ownership_type::owner) {
            return data_.size();
        } else {
            return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        }
    }

    /**
     * @brief Get a const pointer to the underlying data.
     * @return A const pointer to the tensor data.
     */
    [[nodiscard]] constexpr auto data() const {
        if constexpr (OwnershipType == ownership_type::owner) {
            return data_.data();
        } else {
            return data_;
        }
    }

    /**
     * @brief Get a pointer to the underlying data.
     * @return A pointer to the tensor data.
     */
    [[nodiscard]] constexpr auto data() { return const_cast<T *>(std::as_const(*this).data()); }

    /**
     * @brief Get the error checking mode of the tensor.
     * @return The error checking mode.
     */
    static constexpr auto error_checking() { return ErrorChecking; }

    /**
     * @brief Get the ownership type of the tensor.
     * @return The ownership type.
     */
    static constexpr auto ownership_type() { return OwnershipType; }

    /**
     * @brief Get the memory space of the tensor.
     * @return The memory space.
     */
    static constexpr auto memory_space() { return MemorySpace; }

    /**
     * @brief Access an element of the tensor using an index array.
     * @param indices An array or vector of indices.
     * @return A const reference to the element at the specified indices.
     * @throws std::out_of_range If indices are out of bounds (when ErrorChecking is enabled).
     */
    auto access_element(const index_type &indices) const -> const T & {
        if constexpr (ErrorChecking == error_checking::enabled) {
            check_bounds(indices);
        }
        return data()[compute_offset(indices)];
    }

    /**
     * @brief Access an element of the tensor using variadic indices.
     * @param indices Variadic list of indices.
     * @return A const reference to the element at the specified indices.
     */
    template <typename... Indices> auto operator()(Indices... indices) const -> const T & {
        return access_element({static_cast<std::size_t>(indices)...});
    }

    /**
     * @brief Access an element of the tensor using variadic indices (non-const version).
     * @param indices Variadic list of indices.
     * @return A reference to the element at the specified indices.
     */
    template <typename... Indices> auto operator()(Indices... indices) -> T & {
        return const_cast<T &>(std::as_const(*this)(indices...));
    }

#ifndef _MSC_VER
    // MSVC does not support the multidimensional subscript operator yet

    /**
     * @brief Access an element of the tensor using variadic indices.
     * @param indices Variadic list of indices.
     * @return A const reference to the element at the specified indices.
     */
    template <typename... Indices> auto operator[](Indices... indices) const -> const T & {
        return access_element({static_cast<std::size_t>(indices)...});
    }

    /**
     * @brief Access an element of the tensor using variadic indices (non-const version).
     * @param indices Variadic list of indices.
     * @return A reference to the element at the specified indices.
     */
    template <typename... Indices> auto operator[](Indices... indices) -> T & {
        return const_cast<T &>(std::as_const(*this)(indices...));
    }
#endif

    /**
     * @brief Access an element of the tensor using an index array.
     * @param indices An array or vector of indices.
     * @return A const reference to the element at the specified indices.
     */
    auto operator[](const index_type &indices) const -> const T & { return access_element(indices); }

    /**
     * @brief Access an element of the tensor using an index array (non-const version).
     * @param indices An array or vector of indices.
     * @return A reference to the element at the specified indices.
     */
    auto operator[](const index_type &indices) -> T & { return const_cast<T &>(std::as_const(*this)[indices]); }

    /**
     * @brief Create a subview of the tensor with fixed shape.
     * @tparam SubviewShape The shape type of the subview.
     * @param start_indices The starting indices of the subview.
     * @return A new tensor representing the subview.
     * @throws std::invalid_argument If start_indices size doesn't match tensor rank (when ErrorChecking is enabled).
     */
    template <typename SubviewShape>
    auto subview(const index_type &start_indices)
        requires fixed_shape<Shape> && fixed_shape<SubviewShape>
    {
        static_assert(SubviewShape::size() <= Shape::size(),
                      "Subview dimensions must be less than or equal to tensor rank");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (start_indices.size() != rank()) {
                throw std::invalid_argument("Subview start indices must match tensor rank");
            }
        }
        return tensor<T, SubviewShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            data() + compute_offset(start_indices));
    }

    /**
     * @brief Create a const subview of the tensor with fixed shape.
     * @tparam SubviewShape The shape type of the subview.
     * @param start_indices The starting indices of the subview.
     * @return A new const tensor representing the subview.
     * @throws std::invalid_argument If start_indices size doesn't match tensor rank (when ErrorChecking is enabled).
     */
    template <typename SubviewShape>
    auto subview(const index_type &start_indices) const
        requires fixed_shape<Shape> && fixed_shape<SubviewShape>
    {
        static_assert(SubviewShape::size() <= Shape::size(),
                      "Subview dimensions must be less than or equal to tensor rank");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (start_indices.size() != rank()) {
                throw std::invalid_argument("Subview start indices must match tensor rank");
            }
        }
        return tensor<const T, SubviewShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            data() + compute_offset(start_indices));
    }

    /**
     * @brief Create a subview of the tensor with fixed shape using variadic indices.
     * @tparam SubviewShape The shape type of the subview.
     * @tparam Indices Types of the variadic indices.
     * @param start_indices Variadic list of starting indices for the subview.
     * @return A new tensor representing the subview.
     * @throws std::invalid_argument If the number of indices doesn't match tensor rank (when ErrorChecking is enabled).
     */
    template <typename SubviewShape, typename... Indices>
    auto subview(Indices... start_indices)
        requires fixed_shape<Shape> && fixed_shape<SubviewShape>
    {
        static_assert(SubviewShape::size() <= Shape::size(),
                      "Subview dimensions must be less than or equal to tensor rank");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (sizeof...(Indices) != rank()) {
                throw std::invalid_argument("Subview start indices must match tensor rank");
            }
        }
        return tensor<T, SubviewShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            data() + compute_offset({start_indices...}));
    }

    /**
     * @brief Create a const subview of the tensor with fixed shape using variadic indices.
     * @tparam SubviewShape The shape type of the subview.
     * @tparam Indices Types of the variadic indices.
     * @param start_indices Variadic list of starting indices for the subview.
     * @return A new const tensor representing the subview.
     * @throws std::invalid_argument If the number of indices doesn't match tensor rank (when ErrorChecking is enabled).
     */
    template <typename SubviewShape, typename... Indices>
    auto subview(Indices... start_indices) const
        requires fixed_shape<Shape> && fixed_shape<SubviewShape>
    {
        static_assert(SubviewShape::size() <= Shape::size(),
                      "Subview dimensions must be less than or equal to tensor rank");
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (sizeof...(Indices) != rank()) {
                throw std::invalid_argument("Subview start indices must match tensor rank");
            }
        }
        return tensor<const T, SubviewShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
            data() + compute_offset({start_indices...}));
    }

    /**
     * @brief Create a subview of the tensor with dynamic shape.
     * @param subview_shape The shape of the subview.
     * @param start_indices The starting indices of the subview.
     * @return A new tensor representing the subview.
     * @throws std::invalid_argument If subview_shape or start_indices size doesn't match tensor rank (when
     * ErrorChecking is enabled).
     */
    auto subview(const index_type &subview_shape, const index_type &start_indices)
        requires dynamic_shape<Shape> && dynamic_shape<Strides>
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (start_indices.size() > rank() || subview_shape.size() > rank()) {
                throw std::invalid_argument("Subview dimensions must be less than or equal to tensor rank");
            }
        }
        return tensor<T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking, ownership_type::reference,
                      MemorySpace>(data() + compute_offset(start_indices), subview_shape, strides_);
    }

    /**
     * @brief Create a const subview of the tensor with dynamic shape.
     * @param subview_shape The shape of the subview.
     * @param start_indices The starting indices of the subview.
     * @return A new const tensor representing the subview.
     * @throws std::invalid_argument If subview_shape or start_indices size doesn't match tensor rank (when
     * ErrorChecking is enabled).
     */
    auto subview(const index_type &subview_shape, const index_type &start_indices) const
        requires dynamic_shape<Shape> && dynamic_shape<Strides>
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (start_indices.size() > rank() || subview_shape.size() > rank()) {
                throw std::invalid_argument("Subview dimensions must be less than or equal to tensor rank");
            }
        }
        return tensor<const T, std::vector<std::size_t>, std::vector<std::size_t>, ErrorChecking,
                      ownership_type::reference, MemorySpace>(data() + compute_offset(start_indices), subview_shape,
                                                              strides_);
    }

    /**
     * @brief Creates a tensor filled with zeros.
     *
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new tensor filled with zeros.
     */
    static auto zeros(const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        if constexpr (fixed_shape<Shape>) {
            return tensor(); // Default constructor initializes to zero for arithmetic types
        } else {
            return tensor(shape, l); // Uses the constructor we defined earlier
        }
    }

    /**
     * @brief Creates a tensor filled with ones.
     *
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new tensor filled with ones.
     */
    static auto ones(const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        if constexpr (fixed_shape<Shape>) {
            return tensor(T(1));
        } else {
            auto t = tensor(shape, l);
            std::fill(t.data(), t.data() + t.size(), T(1));
            return t;
        }
    }

    /**
     * @brief Creates a tensor filled with a specific value.
     *
     * @param value The value to fill the tensor with.
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new tensor filled with the specified value.
     */
    static auto full(const T &value, const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        if constexpr (fixed_shape<Shape>) {
            return tensor(value);
        } else {
            auto t = tensor(shape, l);
            std::fill(t.data(), t.data() + t.size(), value);
            return t;
        }
    }

    /**
     * @brief Creates a tensor with random values in the specified range.
     *
     * @param min The minimum value for the random numbers.
     * @param max The maximum value for the random numbers.
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new tensor filled with random values.
     */
    static auto random(T min, T max, const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);

        if constexpr (fixed_shape<Shape>) {
            tensor t;
            std::generate(t.data(), t.data() + t.size(), [&]() { return dis(gen); });
            return t;
        } else {
            auto t = tensor(shape, l);
            std::generate(t.data(), t.data() + t.size(), [&]() { return dis(gen); });
            return t;
        }
    }

    /**
     * @brief Creates an identity tensor (square tensor with ones on the main diagonal and zeros elsewhere).
     *
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new identity tensor.
     * @throws std::invalid_argument If the tensor is not square (when ErrorChecking is enabled).
     */
    static auto eye(const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        if constexpr (fixed_shape<Shape>) {
            constexpr auto dims = make_array(Shape{});
            static_assert(dims.size() == 2 && dims[0] == dims[1], "Eye tensor must be square");
            tensor t;
            for (size_t i = 0; i < dims[0]; ++i) {
                t(i, i) = T(1);
            }
            return t;
        } else {
            if constexpr (ErrorChecking == error_checking::enabled) {
                if (shape.size() != 2 || shape[0] != shape[1]) {
                    throw std::invalid_argument("Eye tensor must be square");
                }
            }
            auto t = tensor(shape, l);
            for (size_t i = 0; i < shape[0]; ++i) {
                t(i, i) = T(1);
            }
            return t;
        }
    }

    /**
     * @brief Creates a diagonal tensor with the specified value on the main diagonal.
     *
     * @param value The value to put on the main diagonal.
     * @param shape The shape of the tensor (for dynamic shape only).
     * @param l The memory layout of the tensor (for dynamic shape only, default is column_major).
     * @return A new diagonal tensor.
     * @throws std::invalid_argument If the tensor is not square (when ErrorChecking is enabled).
     */
    static auto diag(const T &value, const std::vector<size_t> &shape = {}, layout l = layout::column_major) {
        if constexpr (fixed_shape<Shape>) {
            constexpr auto dims = make_array(Shape{});
            static_assert(dims.size() == 2 && dims[0] == dims[1], "Diagonal tensor must be square");
            tensor t;
            for (size_t i = 0; i < dims[0]; ++i) {
                t(i, i) = value;
            }
            return t;
        } else {
            if constexpr (ErrorChecking == error_checking::enabled) {
                if (shape.size() != 2 || shape[0] != shape[1]) {
                    throw std::invalid_argument("Diagonal tensor must be square");
                }
            }
            auto t = tensor(shape, l);
            for (size_t i = 0; i < shape[0]; ++i) {
                t(i, i) = value;
            }
            return t;
        }
    }

    /**
     * @brief Creates a non-const view of the entire tensor.
     *
     * @return A non-const tensor object with reference ownership representing the entire tensor.
     */
    auto view() {
        if constexpr (fixed_shape<Shape>) {
            return tensor<T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
        } else {
            return tensor<T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data(), this->shape(), this->strides());
        }
    }

    /**
     * @brief Creates a const view of the entire tensor.
     *
     * @return A const tensor object with reference ownership representing the entire tensor.
     */
    auto view() const {
        if constexpr (fixed_shape<Shape>) {
            return tensor<const T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
        } else {
            return tensor<const T, Shape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data(), this->shape(), this->strides());
        }
    }

    /**
     * @brief Creates a view down the diagonal of the tensor.
     *
     * @return A tensor object with reference ownership representing the diagonal of the tensor.
     * @throws std::invalid_argument If the tensor is not square (when ErrorChecking is enabled).
     */
    auto diag_view() {
        if constexpr (fixed_shape<Shape>) {
            static_assert(all_equal(Shape{}), "Diagonal view is only valid for square matrices");
            constexpr size_t diag_size = std::get<0>(make_array(Shape{}));
            using DiagShape = std::index_sequence<diag_size>;
            using DiagStrides = std::index_sequence<sum(Strides{})>;

            return tensor<T, DiagShape, DiagStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data());
        } else {
            if constexpr (ErrorChecking == error_checking::enabled) {
                bool all_equal = std::all_of(this->shape().begin(), this->shape().end(),
                                             [&](size_t s) { return s == this->shape()[0]; });
                if (!all_equal) {
                    throw std::invalid_argument("Diagonal view is only valid for square matrices");
                }
            }
            std::vector<size_t> diag_shape = {this->shape()[0]};
            std::vector<size_t> diag_strides = std::accumulate(this->strides().begin(), this->strides().end(), 0);

            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data(), diag_shape, diag_strides);
        }
    }

    /**
     * @brief Creates a const view down the diagonal of the tensor.
     *
     * @return A const tensor object with reference ownership representing the diagonal of the tensor.
     * @throws std::invalid_argument If the tensor is not square (when ErrorChecking is enabled).
     */
    auto diag_view() const {
        if constexpr (fixed_shape<Shape>) {
            static_assert(all_equal(Shape{}), "Diagonal view is only valid for square matrices");
            constexpr size_t diag_size = std::get<0>(make_array(Shape{}));
            using DiagShape = std::index_sequence<diag_size>;
            using DiagStrides = std::index_sequence<sum(Strides{})>;

            return tensor<T, DiagShape, DiagStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data());
        } else {
            if constexpr (ErrorChecking == error_checking::enabled) {
                bool all_equal = std::all_of(this->shape().begin(), this->shape().end(),
                                             [&](size_t s) { return s == this->shape()[0]; });
                if (!all_equal) {
                    throw std::invalid_argument("Diagonal view is only valid for square matrices");
                }
            }
            std::vector<size_t> diag_shape = {this->shape()[0]};
            std::vector<size_t> diag_strides = std::accumulate(this->strides().begin(), this->strides().end(), 0);

            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data(), diag_shape, diag_strides);
        }
    }

    /**
     * @brief Reshapes the tensor to new dimensions (column-major version).
     *
     * @tparam NewDims Variadic template pack for new dimensions.
     * @return A new tensor with reference ownership and the reshaped dimensions.
     * @throws std::invalid_argument If the new shape doesn't match the total number of elements (when ErrorChecking is
     * enabled).
     */
    template <size_t... NewDims>
    auto reshape()
        requires(fixed_shape<Shape> && fixed_shape<Strides> && OwnershipType == ownership_type::owner &&
                 std::same_as<column_major_strides<Shape>, Strides>)
    {
        constexpr size_t new_size = (NewDims * ...);
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (new_size != this->size()) {
                throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
            }
        }

        using NewShape = std::index_sequence<NewDims...>;
        using NewStrides = column_major_strides<NewShape>;

        return tensor<T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    }

    /**
     * @brief Reshapes the tensor to new dimensions (const column-major version).
     *
     * @tparam NewDims Variadic template pack for new dimensions.
     * @return A new const tensor with reference ownership and the reshaped dimensions.
     * @throws std::invalid_argument If the new shape doesn't match the total number of elements (when ErrorChecking is
     * enabled).
     */
    template <size_t... NewDims>
    auto reshape() const
        requires(fixed_shape<Shape> && fixed_shape<Strides> && OwnershipType == ownership_type::owner &&
                 std::same_as<column_major_strides<Shape>, Strides>)
    {
        constexpr size_t new_size = (NewDims * ...);
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (new_size != this->size()) {
                throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
            }
        }

        using NewShape = std::index_sequence<NewDims...>;
        using NewStrides = column_major_strides<NewShape>;

        return tensor<const T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data());
    }

    /**
     * @brief Reshapes the tensor to new dimensions (row-major version).
     *
     * @tparam NewDims Variadic template pack for new dimensions.
     * @return A new tensor with reference ownership and the reshaped dimensions.
     * @throws std::invalid_argument If the new shape doesn't match the total number of elements (when ErrorChecking is
     * enabled).
     */
    template <size_t... NewDims>
    auto reshape()
        requires(fixed_shape<Shape> && fixed_shape<Strides> && OwnershipType == ownership_type::owner &&
                 std::same_as<row_major_strides<Shape>, Strides>)
    {
        constexpr size_t new_size = (NewDims * ...);
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (new_size != this->size()) {
                throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
            }
        }

        using NewShape = std::index_sequence<NewDims...>;
        using NewStrides = row_major_strides<NewShape>;

        return tensor<T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(this->data());
    }

    /**
     * @brief Reshapes the tensor to new dimensions (const row-major version).
     *
     * @tparam NewDims Variadic template pack for new dimensions.
     * @return A new const tensor with reference ownership and the reshaped dimensions.
     * @throws std::invalid_argument If the new shape doesn't match the total number of elements (when ErrorChecking is
     * enabled).
     */
    template <size_t... NewDims>
    auto reshape() const
        requires(fixed_shape<Shape> && fixed_shape<Strides> && OwnershipType == ownership_type::owner &&
                 std::same_as<row_major_strides<Shape>, Strides>)
    {
        constexpr size_t new_size = (NewDims * ...);
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (new_size != this->size()) {
                throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
            }
        }

        using NewShape = std::index_sequence<NewDims...>;
        using NewStrides = row_major_strides<NewShape>;

        return tensor<const T, NewShape, NewStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
            this->data());
    }

    /**
     * @brief Returns a flattened view of the tensor.
     *
     * @return A tensor object with reference ownership representing the flattened tensor.
     */
    auto flatten() {
        if constexpr (fixed_shape<Shape>) {
            constexpr size_t total_size = product(Shape{});
            using FlatShape = std::index_sequence<total_size>;
            using FlatStrides = std::index_sequence<1>;

            return tensor<T, FlatShape, FlatStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data());
        } else {
            std::vector<size_t> flat_shape = {this->size()};
            std::vector<size_t> flat_strides = {1};

            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data(), flat_shape, flat_strides);
        }
    }

    /**
     * @brief Returns a flattened view of the tensor (const version).
     *
     * @return A const tensor object with reference ownership representing the flattened tensor.
     */
    auto flatten() const {
        if constexpr (fixed_shape<Shape>) {
            constexpr size_t total_size = product(Shape{});
            using FlatShape = std::index_sequence<total_size>;
            using FlatStrides = std::index_sequence<1>;

            return tensor<const T, FlatShape, FlatStrides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data());
        } else {
            std::vector<size_t> flat_shape = {this->size()};
            std::vector<size_t> flat_strides = {1};

            return tensor<const T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data(), flat_shape, flat_strides);
        }
    }

    /**
     * @brief Returns an iterator of row views.
     *
     * @return An iterator that yields tensor objects with reference ownership for each row.
     * @throws std::invalid_argument If the tensor is not at least 2D (when ErrorChecking is enabled).
     */
    auto rows() {
        if constexpr (fixed_shape<Shape>) {
            using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
            return this->template subviews<RowShape>();
        } else {
            std::vector<std::size_t> row_shape = this->shape();
            row_shape[0] = 1;
            return this->subviews(row_shape);
        }
    }

    /**
     * @brief Returns an iterator of row views (const version).
     *
     * @return An iterator that yields const tensor objects with reference ownership for each row.
     * @throws std::invalid_argument If the tensor is not at least 2D (when ErrorChecking is enabled).
     */
    auto rows() const {
        if constexpr (fixed_shape<Shape>) {
            using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
            return this->template subviews<RowShape>();
        } else {
            std::vector<std::size_t> row_shape = this->shape();
            row_shape[0] = 1;
            return this->subviews(row_shape);
        }
    }

    /**
     * @brief Returns an iterator of column views.
     *
     * @return An iterator that yields tensor objects with reference ownership for each column.
     * @throws std::invalid_argument If the tensor is not at least 2D (when ErrorChecking is enabled).
     */
    auto cols() {
        if constexpr (fixed_shape<Shape>) {
            using ColShape = append_sequence_t<init_sequence_t<Shape>, 1>;
            return this->template subviews<ColShape>();
        } else {
            std::vector<size_t> col_shape = this->shape();
            col_shape[col_shape.size() - 1] = 1;
            return this->subviews(col_shape);
        }
    }

    /**
     * @brief Returns an iterator of column views (const version).
     *
     * @return An iterator that yields const tensor objects with reference ownership for each column.
     * @throws std::invalid_argument If the tensor is not at least 2D (when ErrorChecking is enabled).
     */
    auto cols() const {
        if constexpr (fixed_shape<Shape>) {
            using ColShape = append_sequence_t<init_sequence_t<Shape>, 1>;
            return this->template subviews<ColShape>();
        } else {
            std::vector<size_t> col_shape = this->shape();
            col_shape[col_shape.size() - 1] = 1;
            return this->subviews(col_shape);
        }
    }

    /**
     * @brief Returns a view of a single row.
     *
     * @param index The index of the row to view.
     * @return A tensor object with reference ownership representing the specified row.
     * @throws std::out_of_range If the index is out of bounds (when ErrorChecking is enabled).
     */
    auto row(size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::get<0>(make_array(Shape{}))) {
                throw std::out_of_range("Row index out of range");
            }
        }

        if constexpr (fixed_shape<Shape>) {
            using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
            return tensor<T, RowShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data() + index * std::get<0>(make_array(Strides{})));
        } else {
            std::vector<size_t> row_shape = this->shape();
            row_shape[0] = 1;
            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data() + index * this->strides()[0], row_shape, this->strides());
        }
    }

    /**
     * @brief Returns a const view of a single row.
     *
     * @param index The index of the row to view.
     * @return A const tensor object with reference ownership representing the specified row.
     * @throws std::out_of_range If the index is out of bounds (when ErrorChecking is enabled).
     */
    auto row(size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::get<0>(make_array(Shape{}))) {
                throw std::out_of_range("Row index out of range");
            }
        }

        if constexpr (fixed_shape<Shape>) {
            using RowShape = prepend_sequence_t<tail_sequence_t<Shape>, 1>;
            return tensor<T, RowShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data() + index * std::get<0>(make_array(Strides{})));
        } else {
            std::vector<size_t> row_shape = this->shape();
            row_shape[0] = 1;
            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data() + index * this->strides()[0], row_shape, this->strides());
        }
    }

    /**
     * @brief Returns a view of a single column.
     *
     * @param index The index of the column to view.
     * @return A tensor object with reference ownership representing the specified column.
     * @throws std::out_of_range If the index is out of bounds (when ErrorChecking is enabled).
     */
    auto col(size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::get<1>(make_array(Shape{}))) {
                throw std::out_of_range("Column index out of range");
            }
        }

        if constexpr (fixed_shape<Shape>) {
            using ColShape = append_sequence_t<init_sequence_t<Shape>, 1>;
            return tensor<T, ColShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data() + index * std::get<1>(make_array(Strides{})));
        } else {
            std::vector<size_t> col_shape = this->shape();
            col_shape[col_shape.size() - 1] = 1;
            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data() + index * this->strides()[1], col_shape, this->strides());
        }
    }

    /**
     * @brief Returns a const view of a single column.
     *
     * @param index The index of the column to view.
     * @return A const tensor object with reference ownership representing the specified column.
     * @throws std::out_of_range If the index is out of bounds (when ErrorChecking is enabled).
     */
    auto col(size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= std::get<1>(make_array(Shape{}))) {
                throw std::out_of_range("Column index out of range");
            }
        }

        if constexpr (fixed_shape<Shape>) {
            using ColShape = append_sequence_t<init_sequence_t<Shape>, 1>;
            return tensor<T, ColShape, Strides, ErrorChecking, ownership_type::reference, MemorySpace>(
                this->data() + index * std::get<1>(make_array(Strides{})));
        } else {
            std::vector<size_t> col_shape = this->shape();
            col_shape[col_shape.size() - 1] = 1;
            return tensor<T, std::vector<size_t>, std::vector<size_t>, ErrorChecking, ownership_type::reference,
                          MemorySpace>(this->data() + index * this->strides()[1], col_shape, this->strides());
        }
    }

    /**
     * @brief Reshapes the tensor in-place (only for dynamic shape tensors).
     *
     * @param new_shape The new shape for the tensor.
     * @throws std::invalid_argument If the new shape doesn't match the total number of elements (when ErrorChecking is
     * enabled).
     */
    void reshape(std::vector<size_t> new_shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && dynamic_shape<Strides> && OwnershipType == ownership_type::owner)
    {
        if constexpr (ErrorChecking == error_checking::enabled) {
            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>());
            if (new_size != this->size()) {
                throw std::invalid_argument("New shape must have the same number of elements as the original tensor");
            }
        }

        this->shape_ = std::move(new_shape);
        this->strides_ = compute_strides(l);
    }

    friend auto operator<<(std::ostream &os, const tensor &t) -> std::ostream & {
        // Helper function to print a single element
        auto print_element = [&os](const T &element) {
            os << std::setw(10) << std::setprecision(4) << std::fixed << element;
        };

        // Helper function to print a row
        auto print_row = [&](const auto &row) {
            os << "[";
            bool first = true;
            for (const auto &element : row) {
                if (!first) {
                    os << ", ";
                }
                print_element(element);
                first = false;
            }
            os << "]";
        };

        // Print the tensor shape
        os << "Tensor(shape=[";
        const auto &shape = t.shape();
        for (size_t i = 0; i < t.rank(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            os << shape[i];
        }
        os << "]):\n";

        if (t.rank() == 1) {
            print_row(t);
        } else if (t.rank() == 2) {
            os << "[";
            bool first_row = true;
            for (const auto &row : t.rows()) {
                if (!first_row) {
                    os << ",\n ";
                }
                print_row(row);
                first_row = false;
            }
            os << "]";
        } else {
            // For higher dimensions, we'll print the first two dimensions and indicate if there are more
            os << "[";
            bool first_slice = true;
            for (size_t i = 0; i < std::min(shape[0], size_t(3)); ++i) {
                if (!first_slice) {
                    os << ",\n\n";
                }
                os << " [";
                bool first_row = true;
                for (size_t j = 0; j < std::min(shape[1], size_t(3)); ++j) {
                    if (!first_row) {
                        os << ",\n  ";
                    }
                    if constexpr (fixed_shape<Shape>) {
                        print_row(t.template subview<tail_sequence_t<tail_sequence_t<Shape>>>(i, j));
                    } else {
                        std::vector<size_t> subview_shape(t.shape().begin() + 2, t.shape().end());
                        print_row(t.subview(subview_shape, {i, j}));
                    }
                    first_row = false;
                }
                if (shape[1] > 3) {
                    os << ",\n  ...";
                }
                os << "]";
                first_slice = false;
            }
            if (shape[0] > 3) {
                os << ",\n\n ...";
            }
            os << "]";
        }

        return os;
    }

  private:
    /**
     * @brief Compute the offset for a given set of indices (implementation for fixed strides).
     * @param indices The indices to compute the offset for.
     * @param seq Index sequence for compile-time unrolling.
     * @return The computed offset.
     */
    template <std::size_t... Is>
    [[nodiscard]] constexpr auto compute_offset_impl(const index_type &indices,
                                                     std::index_sequence<Is...> /*unused*/) const -> std::size_t {
        return ((indices[Is] * std::get<Is>(make_array(Strides{}))) + ...);
    }

    /**
     * @brief Compute the offset for a given set of indices.
     * @param indices The indices to compute the offset for.
     * @return The computed offset.
     */
    [[nodiscard]] constexpr auto compute_offset(const index_type &indices) const -> std::size_t {
        if constexpr (fixed_shape<Strides>) {
            return compute_offset_impl(indices, std::make_index_sequence<Strides::size()>{});
        } else {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < rank(); ++i) {
                offset += indices[i] * strides_[i];
            }
            return offset;
        }
    }

    /**
     * @brief Check if the given indices are within bounds.
     * @param indices The indices to check.
     * @throws std::out_of_range If indices are out of bounds.
     */
    constexpr auto check_bounds(const index_type &indices) const -> void {
        if (indices.size() != rank()) {
            throw std::out_of_range("Invalid number of indices");
        }
        for (std::size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape()[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    /**
     * @brief Compute strides for a given layout.
     * @param layout The desired memory layout (row_major or column_major).
     * @return A vector of computed strides.
     */
    [[nodiscard]] auto compute_strides(layout layout) const -> std::vector<std::size_t> {
        std::vector<std::size_t> strides(rank());
        std::size_t stride = 1;
        if (layout == layout::row_major) {
            for (std::size_t i = rank(); i > 0; --i) {
                strides[i - 1] = stride;
                stride *= shape_[i - 1];
            }
        } else {
            for (std::size_t i = 0; i < rank(); ++i) {
                strides[i] = stride;
                stride *= shape_[i];
            }
        }
        return strides;
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_HPP