#ifndef SQUINT_DYNAMIC_TENSOR_HPP
#define SQUINT_DYNAMIC_TENSOR_HPP

#include "squint/iterable_tensor.hpp"
#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <numeric>
#include <random>
#include <vector>

namespace squint {

// Dynamic tensor implementation
template <typename T, error_checking ErrorChecking>
class dynamic_tensor : public iterable_tensor<dynamic_tensor<T, ErrorChecking>, T, ErrorChecking> {
    std::vector<T> data_;
    std::vector<std::size_t> shape_;
    layout layout_;

  public:
    using iterable_tensor<dynamic_tensor<T, ErrorChecking>, T, ErrorChecking>::subviews;
    constexpr dynamic_tensor() = default;
    // virtual destructor
    virtual ~dynamic_tensor() = default;
    dynamic_tensor(std::vector<std::size_t> shape, layout layout = layout::column_major)
        : shape_(std::move(shape)), layout_(layout) {
        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size);
    }
    // Construct from vector of elements
    dynamic_tensor(std::vector<std::size_t> shape, const std::vector<T> &elements, layout layout = layout::column_major)
        : shape_(std::move(shape)), layout_(layout) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (elements.size() != std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>())) {
                throw std::invalid_argument("Number of elements must match total size");
            }
        }
        data_ = elements;
    }
    // Fill the tensor with a single value
    dynamic_tensor(std::vector<std::size_t> shape, const T &value, layout layout = layout::column_major)
        : shape_(std::move(shape)), layout_(layout) {
        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size, value);
    }

    // Construct from a tensor block
    template <tensor BlockTensor>
    dynamic_tensor(std::vector<std::size_t> shape, const BlockTensor &block)
        : shape_(std::move(shape)), layout_(block.get_layout()) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (block.size() == 0) {
                throw std::invalid_argument("Block must have non-zero size");
            }
            // blocks must evenly divide the dimensions of the tensor
            auto min_rank = std::min(shape_.size(), block.shape().size());
            for (std::size_t i = 0; i < min_rank; ++i) {
                if (shape_[i] % block.shape()[i] != 0) {
                    throw std::invalid_argument("Block dimensions must evenly divide tensor dimensions");
                }
            }
        }

        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size);

        auto iter = this->subviews(block.shape()).begin();
        std::size_t num_repeats = total_size / block.size();
        for (std::size_t i = 0; i < num_repeats; ++i) {
            *iter = block;
            ++iter;
        }
    }

    // Construct from a vector of tensor blocks or views
    template <tensor BlockTensor>
    dynamic_tensor(const std::vector<BlockTensor> &blocks)
        : shape_(calculate_new_shape(blocks)), layout_(blocks[0].get_layout()) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (blocks.empty()) {
                throw std::invalid_argument("Cannot construct from empty vector of blocks");
            }
        }

        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size);

        auto iter = this->subviews(blocks[0].shape()).begin();
        for (const auto &block : blocks) {
            if constexpr (ErrorChecking == error_checking::enabled) {
                if (block.size() == 0) {
                    throw std::invalid_argument("Block must have non-zero size");
                }
                if (block.size() != blocks[0].size()) {
                    throw std::invalid_argument("All blocks must have the same size");
                }
                if (block.shape().size() != blocks[0].shape().size()) {
                    throw std::invalid_argument("All blocks must have the same rank");
                }
                if (block.shape() != blocks[0].shape()) {
                    throw std::invalid_argument("All blocks must have the same shape");
                }
                if (block.get_layout() != layout_) {
                    throw std::invalid_argument("All blocks must have the same layout");
                }
            }

            *iter = block;
            ++iter;
        }
    }

    // Copy constructor
    dynamic_tensor(const dynamic_tensor &) = default;

    // Move constructor
    dynamic_tensor(dynamic_tensor &&) noexcept = default;

    // Copy assignment
    dynamic_tensor &operator=(const dynamic_tensor &) = default;

    // Move assignment
    dynamic_tensor &operator=(dynamic_tensor &&) noexcept = default;
    constexpr std::size_t rank() const { return shape_.size(); }
    constexpr std::size_t size() const { return data_.size(); }
    constexpr std::vector<std::size_t> shape() const { return shape_; }
    constexpr layout get_layout() const { return layout_; }
    std::vector<std::size_t> strides() const { return calculate_strides(); }

    T &at_impl(const std::vector<size_t> &indices) { return data_[calculate_index(indices)]; }

    const T &at_impl(const std::vector<size_t> &indices) const { return data_[calculate_index(indices)]; }

    T *data() { return data_.data(); }

    const T *data() const { return data_.data(); }

    void reshape(std::vector<std::size_t> new_shape) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>()) != data_.size()) {
                throw std::invalid_argument("New shape must have the same total size");
            }
        }
        shape_ = std::move(new_shape);
    }

    auto view() { return make_dynamic_tensor_view(*this); }

    auto view() const { return make_dynamic_tensor_view(*this); }

    template <typename... Slices> auto subview(Slices... slices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds({slice{slices.start, slices.size}...});
        }
        return view().subview(slices...);
    }

    template <typename... Slices> auto subview(Slices... slices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds({slice{slices.start, slices.size}...});
        }
        return view().subview(slices...);
    }

    auto subview(const std::vector<slice> &slices) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds(slices);
        }
        return view().subview(slices);
    }

    auto subview(const std::vector<slice> &slices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            this->check_subview_bounds(slices);
        }
        return view().subview(slices);
    }

    static dynamic_tensor zeros(const std::vector<std::size_t> &shape, layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        result.fill(T{});
        return result;
    }

    static dynamic_tensor ones(const std::vector<std::size_t> &shape, layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        result.fill(T{1});
        return result;
    }

    static dynamic_tensor full(const std::vector<std::size_t> &shape, const T &value, layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        result.fill(value);
        return result;
    }

    static dynamic_tensor arange(const std::vector<std::size_t> &shape, T start, T step = T{1},
                                 layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        T value = start;
        for (std::size_t i = 0; i < result.size(); ++i) {
            result.data_[i] = value;
            value += step;
        }
        return result;
    }

    template <tensor OtherTensor>
    static dynamic_tensor diag(const OtherTensor &vector, const std::vector<std::size_t> &shape,
                               layout l = layout::column_major) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (vector.rank() != 1) {
                throw std::invalid_argument("Diagonal vector must be 1D");
            }
            if (vector.size() != *std::min_element(shape.begin(), shape.end())) {
                throw std::invalid_argument("Diagonal vector size must be the minimum of the dimensions");
            }
        }
        dynamic_tensor result(shape, l);
        result.fill(T{});

        std::size_t min_dim = *std::min_element(shape.begin(), shape.end());
        for (std::size_t i = 0; i < min_dim; ++i) {
            std::vector<std::size_t> indices(shape.size(), i);
            result.at_impl(indices) = vector.at_impl({i});
        }

        return result;
    }

    static dynamic_tensor diag(const T &value, const std::vector<std::size_t> &shape, layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        result.fill(T{});

        std::size_t min_dim = *std::min_element(shape.begin(), shape.end());
        for (std::size_t i = 0; i < min_dim; ++i) {
            std::vector<std::size_t> indices(shape.size(), i);
            result.at_impl(indices) = value;
        }

        return result;
    }

    static dynamic_tensor random(const std::vector<std::size_t> &shape, T low = T{}, T high = T{1},
                                 layout l = layout::column_major) {
        dynamic_tensor result(shape, l);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(low, high);

        for (std::size_t i = 0; i < result.size(); ++i) {
            result.data_[i] = dis(gen);
        }

        return result;
    }

    static dynamic_tensor I(const std::vector<std::size_t> &shape, layout l = layout::column_major) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (!std::equal(shape.begin() + 1, shape.end(), shape.begin())) {
                throw std::invalid_argument("All dimensions must be equal for identity tensor");
            }
        }
        dynamic_tensor result(shape, l);
        result.fill(T{});

        std::size_t dim = shape[0];
        for (std::size_t i = 0; i < dim; ++i) {
            std::vector<std::size_t> indices(shape.size(), i);
            result.at_impl(indices) = T{1};
        }

        return result;
    }

    void fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }

    auto flatten() { return dynamic_tensor_view<T, ErrorChecking>(data_.data(), {data_.size()}, {1}, layout_); }

    auto flatten() const {
        return const_dynamic_tensor_view<T, ErrorChecking>(data_.data(), {data_.size()}, {1}, layout_);
    }

    auto rows() {
        if (shape_.empty()) {
            // For 0D tensors
            return this->subviews(std::vector<std::size_t>{});
        }
        std::vector<std::size_t> row_shape = shape_;
        row_shape[0] = 1;
        return this->subviews(row_shape);
    }

    auto cols() {
        if (shape_.size() < 2) {
            // For 0D and 1D tensors
            return this->subviews(shape_);
        }
        std::vector<std::size_t> col_shape = shape_;
        col_shape[shape_.size() - 1] = 1;
        return this->subviews(col_shape);
    }

    auto rows() const {
        if (shape_.empty()) {
            // For 0D tensors
            return this->subviews(std::vector<std::size_t>{});
        }
        std::vector<std::size_t> row_shape = shape_;
        row_shape[0] = 1;
        return this->subviews(row_shape);
    }

    auto cols() const {
        if (shape_.size() < 2) {
            // For 0D and 1D tensors
            return this->subviews(shape_);
        }
        std::vector<std::size_t> col_shape = shape_;
        col_shape[shape_.size() - 1] = 1;
        return this->subviews(col_shape);
    }

    auto row(std::size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= shape_[0]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        std::vector<slice> row_slices(shape_.size());
        row_slices[0] = slice{index, 1};
        for (std::size_t i = 1; i < shape_.size(); ++i) {
            row_slices[i] = slice{0, shape_[i]};
        }
        return this->subview(row_slices);
    }

    auto row(std::size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= shape_[0]) {
                throw std::out_of_range("Row index out of range");
            }
        }
        std::vector<slice> row_slices(shape_.size());
        row_slices[0] = slice{index, 1};
        for (std::size_t i = 1; i < shape_.size(); ++i) {
            row_slices[i] = slice{0, shape_[i]};
        }
        return this->subview(row_slices);
    }

    auto col(std::size_t index) {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= shape_[shape_.size() - 1]) {
                throw std::out_of_range("Column index out of range");
            }
        }
        std::vector<slice> col_slices(shape_.size());
        col_slices[shape_.size() - 1] = slice{index, 1};
        for (std::size_t i = 0; i < shape_.size() - 1; ++i) {
            col_slices[i] = slice{0, shape_[i]};
        }
        return this->subview(col_slices);
    }

    auto col(std::size_t index) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (index >= shape_[shape_.size() - 1]) {
                throw std::out_of_range("Column index out of range");
            }
        }
        std::vector<slice> col_slices(shape_.size());
        col_slices[shape_.size() - 1] = slice{index, 1};
        for (std::size_t i = 0; i < shape_.size() - 1; ++i) {
            col_slices[i] = slice{0, shape_[i]};
        }
        return this->subview(col_slices);
    }

  private:
    size_t calculate_index(const std::vector<size_t> &indices) const {
        if constexpr (ErrorChecking == error_checking::enabled) {
            if (indices.size() != shape_.size()) {
                throw std::invalid_argument("Incorrect number of indices");
            }
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range("Index out of range");
                }
            }
        }

        size_t index = 0;
        size_t stride = 1;

        if (layout_ == layout::row_major) {
            for (int i = shape_.size() - 1; i >= 0; --i) {
                index += indices[i] * stride;
                stride *= shape_[i];
            }
        } else { // column_major
            for (size_t i = 0; i < shape_.size(); ++i) {
                index += indices[i] * stride;
                stride *= shape_[i];
            }
        }

        return index;
    }

    std::vector<std::size_t> calculate_strides() const {
        std::vector<std::size_t> strides(this->shape().size());
        std::size_t stride = 1;
        if (this->get_layout() == layout::row_major) {
            for (int i = this->shape().size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= this->shape()[i];
            }
        } else {
            for (std::size_t i = 0; i < this->shape().size(); ++i) {
                strides[i] = stride;
                stride *= this->shape()[i];
            }
        }
        return strides;
    }

    // Helper function to calculate the new shape for a vector of blocks
    template <tensor BlockTensor>
    std::vector<std::size_t> calculate_new_shape(const std::vector<BlockTensor> &blocks) const {
        if (blocks.empty()) {
            return {};
        }
        // the shape is the shape of the blocks times the number of blocks
        std::vector<std::size_t> new_shape = blocks[0].shape();
        for (auto &dim : new_shape) {
            dim *= blocks.size();
        }

        return new_shape;
    }
};

template <typename T> using tens_t = dynamic_tensor<T, error_checking::disabled>;
using itens = tens_t<int>;
using utens = tens_t<unsigned char>;
using tens = tens_t<float>;
using dtens = tens_t<double>;
using btens = tens_t<bool>;

} // namespace squint

#endif // SQUINT_DYNAMIC_TENSOR_HPP