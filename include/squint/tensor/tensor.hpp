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

#ifndef SQUINT_TENSOR_TENSOR_HPP
#define SQUINT_TENSOR_TENSOR_HPP

#include "squint/core/concepts.hpp"
#include "squint/core/error_checking.hpp"
#include "squint/core/layout.hpp"
#include "squint/core/memory.hpp"
#include "squint/tensor/flat_iterator.hpp"
#include "squint/tensor/subview_iterator.hpp"

#include <array>
#include <cstddef>
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
class tensor {
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
    data_storage data_;

  public:
    using value_type = T;         ///< The type of the tensor elements.
    using shape_type = Shape;     ///< The type used to represent the tensor shape.
    using strides_type = Strides; ///< The type used to represent the tensor strides.
    /// @brief The type used for indexing, std::array for fixed shapes, std::vector for dynamic shapes.
    using index_type =
        std::conditional_t<fixed_shape<Shape>, std::array<std::size_t, Shape::size()>, std::vector<std::size_t>>;

    // Constructors
    tensor()
        requires(OwnershipType == ownership_type::owner)
    = default;
    tensor(std::initializer_list<T> init)
        requires(OwnershipType == ownership_type::owner);
    explicit tensor(const T &value)
        requires(OwnershipType == ownership_type::owner);
    tensor(const std::array<T, product(Shape{})> &elements)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(Shape shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(std::vector<size_t> shape, const std::vector<T> &elements, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(T *data, Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::reference);
    tensor(T *data)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::reference);

    // Accessors
    [[nodiscard]] constexpr auto rank() const -> std::size_t;
    [[nodiscard]] constexpr auto shape() const -> const index_type &;
    [[nodiscard]] constexpr auto strides() const -> const index_type &;
    [[nodiscard]] constexpr auto size() const -> std::size_t;
    [[nodiscard]] constexpr auto data() const -> const T *;
    [[nodiscard]] constexpr auto data() -> T *;

    // Static accessors
    static constexpr auto error_checking() -> error_checking { return ErrorChecking; };
    static constexpr auto ownership_type() -> ownership_type { return OwnershipType; };
    static constexpr auto memory_space() -> memory_space { return MemorySpace; };

    // Element access
    auto access_element(const index_type &indices) const -> const T &;
    template <typename... Indices> auto operator()(Indices... indices) const -> const T &;
    template <typename... Indices> auto operator()(Indices... indices) -> T &;
    auto operator[](const index_type &indices) const -> const T &;
    auto operator[](const index_type &indices) -> T &;
#ifndef _MSC_VER
    // MSVC does not support the multidimensional subscript operator yet
    template <typename... Indices> auto operator[](Indices... indices) const -> const T &;
    template <typename... Indices> auto operator[](Indices... indices) -> T &;
#endif

    // Subview operations
    template <typename SubviewShape>
    auto subview(const index_type &start_indices)
        requires fixed_shape<Shape>;
    template <typename SubviewShape>
    auto subview(const index_type &start_indices) const
        requires fixed_shape<Shape>;
    template <std::size_t... Dims, typename... Indices>
    auto subview(Indices... start_indices)
        requires fixed_shape<Shape>;
    template <std::size_t... Dims, typename... Indices>
    auto subview(Indices... start_indices) const
        requires fixed_shape<Shape>;
    auto subview(const index_type &subview_shape, const index_type &start_indices)
        requires dynamic_shape<Shape>;
    auto subview(const index_type &subview_shape, const index_type &start_indices) const
        requires dynamic_shape<Shape>;

    // View operations
    auto view();
    auto view() const;
    auto diag_view();
    auto diag_view() const;

    // Static creation methods
    static auto zeros(const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);
    static auto ones(const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);
    static auto full(const T &value, const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);
    static auto random(T min, T max, const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);
    static auto eye(const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);
    static auto diag(const T &value, const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);

    // Shape manipulation
    template <size_t... NewDims>
    auto reshape()
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    template <size_t... NewDims>
    auto reshape() const
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    auto flatten()
        requires(OwnershipType == ownership_type::owner);
    auto flatten() const
        requires(OwnershipType == ownership_type::owner);
    void reshape(std::vector<size_t> new_shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    void reshape(std::vector<size_t> new_shape, layout l = layout::column_major) const
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    auto transpose();
    auto transpose() const;

    // Iteration methods
    auto rows();
    auto rows() const;
    auto cols();
    auto cols() const;
    auto row(size_t index);
    auto row(size_t index) const;
    auto col(size_t index);
    auto col(size_t index) const;
    auto begin() -> flat_iterator<tensor>;
    auto end() -> flat_iterator<tensor>;
    auto begin() const -> flat_iterator<const tensor>;
    auto end() const -> flat_iterator<const tensor>;
    auto cbegin() const -> flat_iterator<const tensor>;
    auto cend() const -> flat_iterator<const tensor>;
    template <fixed_shape SubviewShape>
    auto subviews() -> iterator_range<subview_iterator<tensor, SubviewShape>>
        requires fixed_shape<Shape>;
    template <std::size_t... Dims>
    auto subviews() -> iterator_range<subview_iterator<tensor, std::index_sequence<Dims...>>>
        requires fixed_shape<Shape>;
    template <fixed_shape SubviewShape>
    auto subviews() const -> iterator_range<subview_iterator<const tensor, SubviewShape>>
        requires fixed_shape<Shape>;
    template <std::size_t... Dims>
    auto subviews() const -> iterator_range<subview_iterator<const tensor, std::index_sequence<Dims...>>>
        requires fixed_shape<Shape>;
    auto subviews(const std::vector<std::size_t> &subview_shape)
        -> iterator_range<subview_iterator<tensor, std::vector<std::size_t>>>;
    auto subviews(const std::vector<std::size_t> &subview_shape) const
        -> iterator_range<subview_iterator<const tensor, std::vector<std::size_t>>>;

  private:
    template <std::size_t... Is>
    [[nodiscard]] constexpr auto compute_offset_impl(const index_type &indices,
                                                     std::index_sequence<Is...> /*unused*/) const -> std::size_t;
    [[nodiscard]] constexpr auto compute_offset(const index_type &indices) const -> std::size_t;
    constexpr auto check_bounds(const index_type &indices) const -> void;
    [[nodiscard]] auto compute_strides(layout layout) const -> std::vector<std::size_t>;
};

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_HPP