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
 * - Single class policy based design
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
#include "squint/util/sequence_utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef SQUINT_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

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
template <typename T, typename Shape, typename Strides = strides::column_major<Shape>,
          error_checking ErrorChecking = error_checking::disabled, ownership_type OwnershipType = ownership_type::owner,
          memory_space MemorySpace = memory_space::host>
class tensor {
  private:
    [[nodiscard]] static constexpr auto _size() -> std::size_t {
        if constexpr (fixed_shape<Shape>) {
            return product(Shape{});
        } else {
            return 0;
        }
    }
    [[nodiscard]] static constexpr auto _rank() -> std::size_t {
        if constexpr (fixed_shape<Shape>) {
            return Shape::size();
        } else {
            return 0;
        }
    }
    /// @brief Type alias for shape storage, using std::integral_constant for fixed shapes.
    using shape_storage =
        std::conditional_t<fixed_shape<Shape>, std::integral_constant<std::monostate, std::monostate{}>, Shape>;
    using device_shape_storage = std::conditional_t<MemorySpace == memory_space::device, std::size_t *,
                                                    std::integral_constant<std::monostate, std::monostate{}>>;
    /// @brief Type alias for strides storage, using std::integral_constant for fixed strides.
    using strides_storage =
        std::conditional_t<fixed_shape<Strides>, std::integral_constant<std::monostate, std::monostate{}>, Strides>;
    using device_strides_storage = std::conditional_t<MemorySpace == memory_space::device, std::size_t *,
                                                      std::integral_constant<std::monostate, std::monostate{}>>;
    /// @brief Type alias for data storage, using std::array for fixed shapes and std::vector for dynamic shapes.
    using data_storage =
        std::conditional_t<OwnershipType == ownership_type::owner,
                           std::conditional_t<fixed_shape<Shape>, std::array<T, _size()>, std::vector<T>>, T *>;

    // Shape and strides storage
    // These are effectively empty if the shape is fixed. [[no_unique_address]] is used to avoid increasing the size of
    // the tensor class. Note: the order here matters, MSVC does not appear to be able to optimize the size of the
    // tensor if these definitions come last in the classes member list.
    NO_UNIQUE_ADDRESS shape_storage shape_;     ///< Storage for tensor shape.
    NO_UNIQUE_ADDRESS strides_storage strides_; ///< Storage for tensor strides.
    NO_UNIQUE_ADDRESS device_shape_storage device_shape_;
    NO_UNIQUE_ADDRESS device_strides_storage device_strides_;
    data_storage data_;

  public:
    using value_type = T;         ///< The type of the tensor elements.
    using shape_type = Shape;     ///< The type used to represent the tensor shape.
    using strides_type = Strides; ///< The type used to represent the tensor strides.
    /// @brief The type used for indexing, std::array for fixed shapes, std::vector for dynamic shapes.
    using index_type =
        std::conditional_t<fixed_shape<Shape>, std::array<std::size_t, _rank()>, std::vector<std::size_t>>;

    ~tensor() = default;

    // Constructors
    tensor()
        requires(OwnershipType == ownership_type::owner && MemorySpace == memory_space::host)
    = default;
    tensor(const tensor &other)
        requires(OwnershipType == ownership_type::owner && MemorySpace == memory_space::host)
    = default;
    tensor(tensor &&other) noexcept
        requires(OwnershipType == ownership_type::owner && MemorySpace == memory_space::host)
    = default;
    tensor(tensor &&other) noexcept
        requires(OwnershipType == ownership_type::reference && MemorySpace == memory_space::device)
    {
        // move constructor for device tensors
        // NOLINTNEXTLINE
        data_ = other.data_;
        if constexpr (dynamic_shape<Shape>) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
        }
        device_shape_ = other.device_shape_;
        device_strides_ = other.device_strides_;
        other.data_ = nullptr;
        other.device_shape_ = nullptr;
        other.device_strides_ = nullptr;
    }
    // Device constructors for uninitialized data
    tensor()
        requires(fixed_shape<Shape> && MemorySpace == memory_space::device &&
                 OwnershipType == ownership_type::reference)
    {
#ifdef SQUINT_USE_CUDA
        cudaError_t malloc_status = cudaMalloc(&data_, _size() * sizeof(T));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor data");
        }
        malloc_status = cudaMalloc(&device_shape_, _rank() * sizeof(std::size_t));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor shape");
        }
        malloc_status = cudaMalloc(&device_strides_, _rank() * sizeof(std::size_t));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor strides");
        }
        auto host_shape = this->shape();
        auto host_strides = make_array(strides::column_major<Shape>{});
        cudaError_t memcpy_status =
            cudaMemcpy(device_shape_, host_shape.data(), _rank() * sizeof(std::size_t), cudaMemcpyHostToDevice);
        if (memcpy_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor shape to device");
        }
        memcpy_status =
            cudaMemcpy(device_strides_, host_strides.data(), _rank() * sizeof(std::size_t), cudaMemcpyHostToDevice);
        if (memcpy_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor strides to device");
        }
#endif
    }
    tensor(const std::vector<size_t>& shape)
        requires(dynamic_shape<Shape> && MemorySpace == memory_space::device &&
                 OwnershipType == ownership_type::reference)
    {
#ifdef SQUINT_USE_CUDA
        // NOLINTNEXTLINE
        shape_ = shape;
        // NOLINTNEXTLINE
        strides_ = compute_strides(layout::column_major, shape);
        cudaError_t malloc_status = cudaMalloc(&data_, size() * sizeof(T));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor data");
        }
        malloc_status = cudaMalloc(&device_shape_, rank() * sizeof(std::size_t));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor shape");
        }
        malloc_status = cudaMalloc(&device_strides_, rank() * sizeof(std::size_t));
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for tensor strides");
        }
        cudaError_t memcpy_status =
            cudaMemcpy(device_shape_, shape.data(), rank() * sizeof(std::size_t), cudaMemcpyHostToDevice);
        if (memcpy_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor shape to device");
        }
        memcpy_status =
            cudaMemcpy(device_strides_, strides_.data(), rank() * sizeof(std::size_t), cudaMemcpyHostToDevice);
        if (memcpy_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor strides to device");
        }
#endif
    }
    // Fixed shape constructors
    tensor(std::initializer_list<T> init)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    explicit tensor(const T &value)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(const std::array<T, _size()> &elements)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    template <fixed_tensor... OtherTensor>
    tensor(const OtherTensor &...ts)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::owner);
    // Dynamic shape constructors
    tensor(Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(Shape shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(std::vector<size_t> shape, const std::vector<T> &elements, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    tensor(std::vector<size_t> shape, const T &value, layout l = layout::column_major)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::owner);
    // Conversion constructors
    template <typename U, typename OtherShape, typename OtherStrides>
    tensor(const tensor<U, OtherShape, OtherStrides, ErrorChecking, OwnershipType, MemorySpace> &other)
        requires(fixed_shape<Shape> && MemorySpace == memory_space::host);
    template <typename U, typename OtherShape, typename OtherStrides>
    tensor(const tensor<U, OtherShape, OtherStrides, ErrorChecking, ownership_type::reference, MemorySpace> &other)
        requires(OwnershipType == ownership_type::owner);
    // Views
    tensor(T *data, Shape shape, Strides strides)
        requires(dynamic_shape<Shape> && OwnershipType == ownership_type::reference);
    tensor(T *data)
        requires(fixed_shape<Shape> && OwnershipType == ownership_type::reference);

    // Destructor
    ~tensor()
        requires(MemorySpace == memory_space::device && OwnershipType == ownership_type::reference)
    {
#ifdef SQUINT_USE_CUDA
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        if (device_shape_ != nullptr) {
            cudaFree(device_shape_);
        }
        if (device_strides_ != nullptr) {
            cudaFree(device_strides_);
        }
#endif
    }

    // Assignment operators
    template <typename U, typename OtherShape, typename OtherStrides, ownership_type OtherOwnershipType>
    auto operator=(const tensor<U, OtherShape, OtherStrides, ErrorChecking, OtherOwnershipType, MemorySpace> &other)
        -> tensor &;
    auto operator=(const tensor &other) -> tensor &;
    auto operator=(tensor &&other) noexcept -> tensor & = default;

    // Accessors
    [[nodiscard]] constexpr auto rank() const -> std::size_t;
    [[nodiscard]] constexpr auto shape() const -> const index_type &;
    [[nodiscard]] constexpr auto strides() const -> const index_type &;
    [[nodiscard]] constexpr auto size() const -> std::size_t;
    [[nodiscard]] constexpr auto data() const -> const T *;
    [[nodiscard]] constexpr auto data() -> T *;

    // Device Accessors
    [[nodiscard]] auto device_shape() const -> device_shape_storage
        requires(MemorySpace == memory_space::device)
    {
        return device_shape_;
    }
    [[nodiscard]] auto device_strides() const -> device_strides_storage
        requires(MemorySpace == memory_space::device)
    {
        return device_strides_;
    }

    // Static accessors
    static constexpr auto error_checking() -> error_checking { return ErrorChecking; };
    static constexpr auto ownership() -> ownership_type { return OwnershipType; };
    static constexpr auto get_memory_space() -> memory_space { return MemorySpace; };

    // Element access
    auto access_element(const index_type &indices) const -> const T &requires(MemorySpace == memory_space::host);
    template <typename... Indices>
    auto operator()(Indices... indices) const -> const T &requires(MemorySpace == memory_space::host);
    template <typename... Indices>
    auto operator()(Indices... indices) -> T &requires(MemorySpace == memory_space::host);
    auto operator[](const index_type &indices) const -> const T &requires(MemorySpace == memory_space::host);
    auto operator[](const index_type &indices) -> T &requires(MemorySpace == memory_space::host);
#ifndef _MSC_VER
    // MSVC does not support the multidimensional subscript operator yet
    template <typename... Indices>
    auto operator[](Indices... indices) const -> const T &requires(MemorySpace == memory_space::host);
    template <typename... Indices>
    auto operator[](Indices... indices) -> T &requires(MemorySpace == memory_space::host);
#endif

    /**
     * @brief Create an owning copy of the tensor.
     *
     * This method creates an owning copy of the tensor. If the tensor is already an owner, it will return a copy of
     * itself. If the tensor is a reference, it will create a new tensor that owns its data.
     * The resulting tensor will have the same shape and values as the original tensor, but will own its data, use the
     * host memory space, and have column-major strides.
     */
    auto copy() const -> auto
        requires(MemorySpace == memory_space::host)
    {
        if constexpr (fixed_shape<Shape>) {
            using owning_type = tensor<std::remove_const_t<T>, Shape, strides::column_major<Shape>, ErrorChecking,
                                       ownership_type::owner, memory_space::host>;
            return owning_type(*this);
        } else {
            using owning_type = tensor<std::remove_const_t<T>, std::vector<size_t>, std::vector<size_t>, ErrorChecking,
                                       ownership_type::owner, memory_space::host>;
            return owning_type(*this);
        }
    }
#ifdef SQUINT_USE_CUDA
    /**
     * @brief Create a copy of the tensor on the device.
     *
     * This method creates a copy of the tensor on the device and returns a reference tensor to the device memory.
     */
    auto copy() const -> auto
        requires(OwnershipType == ownership_type::reference && MemorySpace == memory_space::device)
    {
        using device_tensor_type = tensor<std::remove_const_t<T>, Shape, Strides, ErrorChecking,
                                          ownership_type::reference, memory_space::device>;
        size_t size = this->size() * sizeof(T);
        // Create device pointer
        void *device_ptr = nullptr;

        // Allocate memory on the device
        cudaError_t malloc_status = cudaMalloc(&device_ptr, size);
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }

        // Copy data from device to device
        cudaError_t memcpy_status =
            cudaMemcpy(device_ptr, static_cast<void *>(const_cast<std::remove_const_t<T> *>(this->data())), size,
                       cudaMemcpyDeviceToDevice);
        if (memcpy_status != cudaSuccess) {
            cudaFree(device_ptr);
            throw std::runtime_error("Failed to copy data to device");
        }

        // Create and return the device tensor
        if constexpr (dynamic_shape<Shape>) {
            return device_tensor_type(static_cast<std::remove_const_t<T> *>(device_ptr), this->shape_, this->strides_);
        } else {
            return device_tensor_type(static_cast<std::remove_const_t<T> *>(device_ptr));
        }
    }

    auto to_device() const -> tensor<std::remove_const_t<T>, Shape, Strides, ErrorChecking, ownership_type::reference,
                                     memory_space::device>
        requires(MemorySpace == memory_space::host && OwnershipType == ownership_type::owner)
    {
        using device_tensor_type = tensor<std::remove_const_t<T>, Shape, Strides, ErrorChecking,
                                          ownership_type::reference, memory_space::device>;
        size_t size = this->size() * sizeof(T);

        // Create device pointer
        void *device_ptr = nullptr;

        // Allocate memory on the device
        cudaError_t malloc_status = cudaMalloc(&device_ptr, size);
        if (malloc_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }

        // Copy data from host to device
        cudaError_t memcpy_status =
            cudaMemcpy(device_ptr, static_cast<void *>(const_cast<std::remove_const_t<T> *>(this->data())), size,
                       cudaMemcpyHostToDevice);
        if (memcpy_status != cudaSuccess) {
            cudaFree(device_ptr);
            throw std::runtime_error("Failed to copy data to device");
        }

        // Create and return the device tensor
        if constexpr (dynamic_shape<Shape>) {
            if constexpr (ErrorChecking == error_checking::enabled) {
                auto column_major_strides = this->compute_strides(layout::column_major, this->shape());
                auto strides = this->strides();
                if (!std::equal(strides.begin(), strides.end(), column_major_strides.begin())) {
                    cudaFree(device_ptr);
                    throw std::runtime_error("Only column-major strides are supported for device tensors");
                }
            }
            return device_tensor_type(static_cast<std::remove_const_t<T> *>(device_ptr), this->shape_, this->strides_);
        } else {
            static_assert(is_column_major_v<Strides, Shape>,
                          "Only column-major strides are supported for device tensors");
            return device_tensor_type(static_cast<std::remove_const_t<T> *>(device_ptr));
        }
    }

    auto to_host() const -> tensor<T, Shape, Strides, ErrorChecking, ownership_type::owner, memory_space::host>
        requires(MemorySpace == memory_space::device && OwnershipType == ownership_type::reference)
    {
        using host_tensor_type = tensor<T, Shape, Strides, ErrorChecking, ownership_type::owner, memory_space::host>;
        size_t size = this->size() * sizeof(T);

        if constexpr (dynamic_shape<Shape>) {
            host_tensor_type host_tensor(this->shape());
            // Copy data from device to host
            cudaError_t memcpy_status =
                cudaMemcpy(static_cast<void *>(host_tensor.data()), static_cast<const void *>(this->data()), size,
                           cudaMemcpyDeviceToHost);
            if (memcpy_status != cudaSuccess) {
                throw std::runtime_error("Failed to copy data to host error code: " + std::to_string(memcpy_status));
            }
            return host_tensor;
        } else {
            host_tensor_type host_tensor{};
            // Copy data from device to host
            cudaError_t memcpy_status =
                cudaMemcpy(static_cast<void *>(host_tensor.data()), static_cast<const void *>(this->data()), size,
                           cudaMemcpyDeviceToHost);
            if (memcpy_status != cudaSuccess) {
                throw std::runtime_error("Failed to copy data to host error code: " + std::to_string(memcpy_status));
            }
            return host_tensor;
        }
    }
#endif

    // Subview operations
    template <typename SubviewShape, typename StepSizes>
    auto subview(const index_type &start_indices)
        requires fixed_shape<Shape>;
    template <typename SubviewShape, typename StepSizes>
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
    auto subview(const index_type &subview_shape, const index_type &start_indices, const index_type &step_sizes)
        requires dynamic_shape<Shape>;
    auto subview(const index_type &subview_shape, const index_type &start_indices, const index_type &step_sizes) const
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
    static auto arange(T start, T step, const std::vector<size_t> &shape = {}, layout l = layout::column_major)
        requires(OwnershipType == ownership_type::owner);

    // Shape manipulation
    template <size_t... NewDims>
    auto reshape()
        requires(fixed_shape<Shape> && fixed_contiguous_strides<Strides, Shape>);
    template <size_t... NewDims>
    auto reshape() const
        requires(fixed_shape<Shape> && fixed_contiguous_strides<Strides, Shape>);
    template <typename NewShape>
    auto reshape()
        requires(fixed_shape<Shape> && fixed_contiguous_strides<Strides, Shape>);
    template <typename NewShape>
    auto reshape() const
        requires(fixed_shape<Shape> && fixed_contiguous_strides<Strides, Shape>);
    auto flatten();
    auto flatten() const;
    auto reshape(std::vector<size_t> new_shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape>);
    auto reshape(std::vector<size_t> new_shape, layout l = layout::column_major) const
        requires(dynamic_shape<Shape>);
    auto set_shape(const std::vector<size_t> &new_shape, layout l = layout::column_major)
        requires(dynamic_shape<Shape>);
    template <valid_index_permutation IndexPermutation>
    auto permute()
        requires fixed_shape<Shape>;
    template <valid_index_permutation IndexPermutation>
    auto permute() const
        requires fixed_shape<Shape>;
    template <std::size_t... Permutation>
    auto permute()
        requires(sizeof...(Permutation) > 0 && valid_index_permutation<std::index_sequence<Permutation...>> &&
                 fixed_shape<Shape>)
    {
        return permute<std::index_sequence<Permutation...>>();
    }
    template <std::size_t... Permutation>
    auto permute() const
        requires(sizeof...(Permutation) > 0 && valid_index_permutation<std::index_sequence<Permutation...>> &&
                 fixed_shape<Shape>)
    {
        return permute<std::index_sequence<Permutation...>>();
    }
    auto permute(const std::vector<std::size_t> &index_permutation)
        requires dynamic_shape<Shape>;
    auto permute(const std::vector<std::size_t> &index_permutation) const
        requires dynamic_shape<Shape>;
    auto transpose();
    auto transpose() const;

    // Iteration methods
    auto rows()
        requires(MemorySpace == memory_space::host);
    auto rows() const
        requires(MemorySpace == memory_space::host);
    auto cols()
        requires(MemorySpace == memory_space::host);
    auto cols() const
        requires(MemorySpace == memory_space::host);
    auto row(size_t index)
        requires(MemorySpace == memory_space::host);
    auto row(size_t index) const
        requires(MemorySpace == memory_space::host);
    auto col(size_t index)
        requires(MemorySpace == memory_space::host);
    auto col(size_t index) const
        requires(MemorySpace == memory_space::host);
    auto begin() -> flat_iterator<tensor>
        requires(MemorySpace == memory_space::host);
    auto end() -> flat_iterator<tensor>
        requires(MemorySpace == memory_space::host);
    auto begin() const -> flat_iterator<const tensor>
        requires(MemorySpace == memory_space::host);
    auto end() const -> flat_iterator<const tensor>
        requires(MemorySpace == memory_space::host);
    auto cbegin() const -> flat_iterator<const tensor>
        requires(MemorySpace == memory_space::host);
    auto cend() const -> flat_iterator<const tensor>
        requires(MemorySpace == memory_space::host);
    template <fixed_shape SubviewShape>
    auto subviews() -> iterator_range<subview_iterator<tensor, SubviewShape>>
        requires(fixed_shape<Shape> && MemorySpace == memory_space::host);
    template <std::size_t... Dims>
    auto subviews() -> iterator_range<subview_iterator<tensor, std::index_sequence<Dims...>>>
        requires(fixed_shape<Shape> && MemorySpace == memory_space::host);
    template <fixed_shape SubviewShape>
    auto subviews() const -> iterator_range<subview_iterator<const tensor, SubviewShape>>
        requires(fixed_shape<Shape> && MemorySpace == memory_space::host);
    template <std::size_t... Dims>
    auto subviews() const -> iterator_range<subview_iterator<const tensor, std::index_sequence<Dims...>>>
        requires(fixed_shape<Shape> && MemorySpace == memory_space::host);
    auto subviews(const std::vector<std::size_t> &subview_shape)
        -> iterator_range<subview_iterator<tensor, std::vector<std::size_t>>>
        requires(dynamic_shape<Shape> && MemorySpace == memory_space::host);
    auto subviews(const std::vector<std::size_t> &subview_shape) const
        -> iterator_range<subview_iterator<const tensor, std::vector<std::size_t>>>
        requires(dynamic_shape<Shape> && MemorySpace == memory_space::host);

    // Incremental operators
    template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
              enum ownership_type OtherOwnershipType>
    auto
    operator+=(const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other)
        -> tensor &;
    template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
              enum ownership_type OtherOwnershipType>
    auto
    operator-=(const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other)
        -> tensor &;
    // Comparison operators
    template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
              enum ownership_type OtherOwnershipType>
    auto
    operator==(const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other)
        const -> tensor<std::uint8_t, Shape, Strides, ErrorChecking, ownership_type::owner, MemorySpace>;
    template <typename U, typename OtherShape, typename OtherStrides, enum error_checking OtherErrorChecking,
              enum ownership_type OtherOwnershipType>
    auto
    operator!=(const tensor<U, OtherShape, OtherStrides, OtherErrorChecking, OtherOwnershipType, MemorySpace> &other)
        const -> tensor<std::uint8_t, Shape, Strides, ErrorChecking, ownership_type::owner, MemorySpace>;
    // Unary operators
    auto operator-() const -> tensor;
    // scalar operations
    template <dimensionless_scalar U> auto operator*=(const U &s) -> tensor &;
    template <dimensionless_scalar U> auto operator/=(const U &s) -> tensor &;

    // util methods
    [[nodiscard]] auto is_contiguous() const -> bool {
        if constexpr (fixed_shape<Strides>) {
            return (implicit_convertible_strides_v<Strides, strides::row_major<Shape>> ||
                    implicit_convertible_strides_v<Strides, strides::column_major<Shape>>);
        } else {
            return strides() == compute_strides(layout::row_major) ||
                   strides() == compute_strides(layout::column_major);
        }
    }

  private:
    template <std::size_t... Is>
    [[nodiscard]] constexpr auto compute_offset_impl(const index_type &indices,
                                                     std::index_sequence<Is...> /*unused*/) const -> std::size_t;
    [[nodiscard]] constexpr auto compute_offset(const index_type &indices) const -> std::size_t;
    constexpr auto check_bounds(const index_type &indices) const -> void;
    [[nodiscard]] auto compute_strides(layout l) const -> std::vector<std::size_t>
        requires dynamic_shape<Shape>
    {
        return compute_strides(l, shape());
    }
    [[nodiscard]] static auto compute_strides(layout l, const std::vector<std::size_t> &shape) {
        auto rank = shape.size();
        if (l == layout::row_major) {
            // Compute row-major strides runtime
            std::vector<std::size_t> row_major_strides(rank);
            row_major_strides[rank - 1] = 1;
            for (std::size_t i = rank - 1; i > 0; --i) {
                row_major_strides[i - 1] = row_major_strides[i] * shape[i];
            }
            return row_major_strides;
        }
        // Compute column-major strides runtime
        std::vector<std::size_t> column_major_strides(rank);
        column_major_strides[0] = 1;
        for (std::size_t i = 1; i < rank; ++i) {
            column_major_strides[i] = column_major_strides[i - 1] * shape[i - 1];
        }
        return column_major_strides;
    }
};

} // namespace squint

#endif // SQUINT_TENSOR_TENSOR_HPP