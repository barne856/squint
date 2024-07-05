#ifndef SQUINT_DYNAMIC_TENSOR_HPP
#define SQUINT_DYNAMIC_TENSOR_HPP

#include "squint/tensor_base.hpp"
#include "squint/tensor_view.hpp"
#include <numeric>
#include <vector>

namespace squint {

// Dynamic tensor implementation
template <typename T> class dynamic_tensor : public tensor_base<dynamic_tensor<T>, T> {
    std::vector<T> data_;
    std::vector<std::size_t> shape_;
    layout layout_;

  public:
    dynamic_tensor(std::vector<std::size_t> shape, layout layout = layout::column_major)
        : shape_(std::move(shape)), layout_(layout) {
        std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>());
        data_.resize(total_size);
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

    T &at_impl(const std::vector<size_t> &indices) { return data_[calculate_index(indices)]; }

    const T &at_impl(const std::vector<size_t> &indices) const { return data_[calculate_index(indices)]; }

    T *data() { return data_.data(); }

    const T *data() const { return data_.data(); }

    void reshape(std::vector<std::size_t> new_shape) {
        if (std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<>()) != data_.size()) {
            throw std::invalid_argument("New shape must have the same total size");
        }
        shape_ = std::move(new_shape);
    }

    dynamic_tensor_view<T> view() {
        return dynamic_tensor_view<T>(this->data(), this->shape(), calculate_strides(), this->get_layout());
    }

    dynamic_tensor_view<const T> view() const {
        return dynamic_tensor_view<const T>(this->data(), this->shape(), calculate_strides(), this->get_layout());
    }

  private:
    size_t calculate_index(const std::vector<size_t> &indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Incorrect number of indices");
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
};

// Deduction guide
template <typename T, typename... Args> dynamic_tensor(T, Args...) -> dynamic_tensor<T>;

} // namespace squint

#endif // SQUINT_DYNAMIC_TENSOR_HPP