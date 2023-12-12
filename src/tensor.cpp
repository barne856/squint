/**
 * @file tensor.hpp
 * @author Brendan Barnes
 * @brief Implementation of tensor data structures
 *
 * @copyright Copyright (c) 2022
 *
 */
module;
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

export module squint:tensor;

import :dimension;

export namespace squint {

// Utility -------------------------------------------------------------------------------------------------------
// constant values -----------------------------------------------------------------------------------------------------
inline constexpr int dynamic_shape = -1;
inline constexpr int success = 0;
// forward declarations ------------------------------------------------------------------------------------------------
template <class U, typename T, int... sizes> class tensor_base;
template <typename T, int... sizes> class tensor_ref;
template <typename T, int... sizes> class tensor;
// type traits ---------------------------------------------------------------------------------------------------------
// any fixed or dynamic sized tensor
template <typename T> struct is_tensor {
    static constexpr bool value = false;
};
template <template <typename, int...> class U, typename T, int... sizes> struct is_tensor<U<T, sizes...>> {
    static constexpr bool value = std::is_base_of<tensor_base<U<T, sizes...>, T, sizes...>, U<T, sizes...>>::value;
};
// dynamic sized tensor only
template <typename T> struct is_dynamic_tensor {
    static constexpr bool value = false;
};
template <template <typename, int...> class U, typename T> struct is_dynamic_tensor<U<T, dynamic_shape>> {
    static constexpr bool value =
        std::is_base_of<tensor_base<U<T, dynamic_shape>, T, dynamic_shape>, U<T, dynamic_shape>>::value;
};
// get the specific (non-ref) tensor type of a generic tensor T
template <typename T> struct tensor_type {};
template <template <typename, int...> class U, typename T, int... sizes> struct tensor_type<U<T, sizes...>> {
    using type = tensor<T, sizes...>;
};
// compile-time list functions -----------------------------------------------------------------------------------------
// checks if a compile-time list of indices has no duplicates
template <size_t... indices> constexpr bool is_unique() {
    std::array<size_t, sizeof...(indices)> arr{indices...};
    return static_cast<bool>(std::unique(arr.begin(), arr.end()) == arr.end());
}
// get the index of the first template parameter that is not equal to 1
template <int... sizes> constexpr size_t get_first_not_one() {
    constexpr size_t N = sizeof...(sizes);
    std::array<int, N> shape{sizes...};
    for (int i = 0; i < static_cast<int>(N); i++) {
        if (shape[i] != 1) {
            return static_cast<size_t>(i);
        }
    }
    return N; // return N if all are eqaul to 1
}
// create a tuple from a compile-time list of integers that has the first value not equal to 1 set to 1
template <int... sizes> constexpr auto index() {
    auto tuple = std::make_tuple(sizes...);
    constexpr size_t first_not_one = get_first_not_one<sizes...>();
    std::get<first_not_one>(tuple) = 1;
    return tuple;
}
// create a tuple from a compile-time list of integers which has the values reversed
template <int... sizes> constexpr auto reverse() {
    auto t = std::make_tuple(sizes...);
    constexpr size_t N = sizeof...(sizes);
    return [&t]<auto... I>(std::index_sequence<I...>) {
        return std::tuple{std::get<N - 1 - I>(t)...};
    }(std::make_index_sequence<N>{});
}
// create a tuple from a compile-time list of integers with all leading 1s removed
template <int... sizes> constexpr auto remove_leading() {
    constexpr auto t = std::make_tuple(sizes...);
    constexpr size_t first_not_one = get_first_not_one<sizes...>();
    constexpr size_t N = sizeof...(sizes) - first_not_one;
    return [&t]<auto... I>(std::index_sequence<I...>) {
        return std::tuple{std::get<I + first_not_one>(t)...};
    }(std::make_index_sequence<N>{});
}
// create a tuple from a compile-time list of integers with all trailing 1s removed
template <int... sizes> constexpr auto remove_trailing() {
    constexpr size_t M = sizeof...(sizes);
    // reverse
    constexpr auto t1 = reverse<sizes...>();
    // remove leading
    constexpr auto t2 = [&]<auto... I>(std::index_sequence<I...>) {
        return remove_leading<std::get<I>(t1)...>();
    }(std::make_index_sequence<M>{});
    constexpr size_t N = std::tuple_size<decltype(t2)>::value;
    // reverse back
    return [&]<auto... J>(std::index_sequence<J...>) {
        return reverse<std::get<J>(t2)...>();
    }(std::make_index_sequence<N>{});
}
// create an array of integers with all trailing 1s removed
template <int... sizes> constexpr auto squeeze_array() {
    constexpr size_t M = sizeof...(sizes);
    // reverse
    constexpr auto t1 = reverse<sizes...>();
    // remove leading
    constexpr auto t2 = [&]<auto... I>(std::index_sequence<I...>) {
        return remove_leading<std::get<I>(t1)...>();
    }(std::make_index_sequence<M>{});
    constexpr size_t N = std::tuple_size<decltype(t2)>::value;
    // reverse back
    constexpr auto t3 = [&]<auto... J>(std::index_sequence<J...>) {
        return reverse<std::get<J>(t2)...>();
    }(std::make_index_sequence<N>{});
    return [&]<auto... J>(std::index_sequence<J...>) {
        return std::array<size_t, sizeof...(J)>{std::get<J>(t3)...};
    }(std::make_index_sequence<N>{});
}
// create a tuple from compile-time list of integers with all leading and trailing 1s removed
template <int... sizes> constexpr auto trim() {
    constexpr size_t N = std::tuple_size<decltype(remove_leading<sizes...>())>::value;
    // remove leading and trailing 1s from tuple
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr auto t1 = remove_leading<sizes...>();
        return remove_trailing<std::get<I>(t1)...>();
    }(std::make_index_sequence<N>{});
}
// concepts ------------------------------------------------------------------------------------------------------------
template <class U>
concept tensorial = is_tensor<U>::value;
template <class U>
concept fixed_tensor = tensorial<U> && !is_dynamic_tensor<U>::value;
template <class U>
concept dynamic_tensor = tensorial<U> && is_dynamic_tensor<U>::value;
template <class U, int... sizes>
concept tensor_shape = std::derived_from<U, tensor_base<U, typename U::value_type, sizes...>>;
// quantity concept
// a quantity is a number that has a dimension
template <class Q>
concept quantitative = std::is_arithmetic<typename Q::value_type>::value && dimensional<typename Q::dimension_type>;
// scalar
template <class U>
concept scalar = tensor_shape<U> || std::is_arithmetic<U>::value || quantitative<U>;
template <class U, typename T>
concept tensor_underlying_type = tensorial<U> && (std::is_same<T, typename U::value_type>::value ||
                                                  std::is_same<T, typename U::value_type::value_type>::value);
// Concept for valid index permutation used to transpose a tensor.
// All indices must have no duplicates, and be less than the total number of indices.
template <size_t... index_permutation>
concept valid_index_permutation =
    is_unique<index_permutation...>() && ((index_permutation < sizeof...(index_permutation)) && ...);
// tensor functions ----------------------------------------------------------------------------------------------------
// Index a tensor to get another tensor n orders lower where n is the number of indices supplied.
// The first n shape sizes != 1 are set to 1
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto index_tensor(T &tens, int i, int j, std::integral auto... other_indices) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr auto tuple = index<T::shape()[I]...>();
        std::array<size_t, sizeof...(I)> offsets{};
        constexpr size_t first_not_one = get_first_not_one<T::shape()[I]...>();
        offsets[first_not_one] = i;
        auto tens_ref = tens.template at<std::get<I>(tuple)...>(offsets);
        return index_tensor(tens_ref, j, other_indices...);
    }(std::make_index_sequence<N>{});
}
// overload for index tensor in the case of a singe index
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto index_tensor(T &tens, int i) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr auto tuple = index<T::shape(I)...>();
        std::array<size_t, sizeof...(I)> offsets{};
        constexpr size_t first_not_one = get_first_not_one<T::shape(I)...>();
        offsets[first_not_one] = i;
        auto tens_ref = tens.template at<std::get<I>(tuple)...>(offsets);
        return tens_ref.remove_trailing();
    }(std::make_index_sequence<N>{});
}
// const overload for index tensor function
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto index_tensor(const T &tens, int i, int j, std::integral auto... other_indices) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr auto tuple = index<T::shape(I)...>();
        std::array<size_t, sizeof...(I)> offsets{};
        constexpr size_t first_not_one = get_first_not_one<T::shape(I)...>();
        offsets[first_not_one] = i;
        auto tens_ref = tens.template at<std::get<I>(tuple)...>(offsets);
        return index_tensor(tens_ref, j, other_indices...);
    }(std::make_index_sequence<N>{});
}
// const overload for index tensor function with a single index
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto index_tensor(const T &tens, int i) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr auto tuple = index<T::shape(I)...>();
        std::array<size_t, sizeof...(I)> offsets{};
        constexpr size_t first_not_one = get_first_not_one<T::shape(I)...>();
        offsets[first_not_one] = i;
        auto tens_ref = tens.template at<std::get<I>(tuple)...>(offsets);
        return tens_ref.remove_trailing();
    }(std::make_index_sequence<N>{});
}
// Demote a tensor (reduce its rank by 1) by truncating the last dimension and slicing at the given index
template <fixed_tensor T>
    requires(!scalar<T>)
auto demote_fixed_tensor(size_t index, T &tens) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr size_t M = N - 1;
        return [&]<auto... J>(std::index_sequence<J...>) {
            constexpr auto tuple = std::make_tuple(T::shape()[I]...);
            std::array<size_t, sizeof...(I)> offsets{};
            offsets[offsets.size() - 1] = index;
            return tens.template at<std::get<J>(tuple)...>(offsets);
        }(std::make_index_sequence<M>{});
    }(std::make_index_sequence<N>{});
}
// const overload for demote tensor function
template <fixed_tensor T>
    requires(!scalar<T>)
auto demote_fixed_tensor(size_t index, const T &tens) {
    constexpr size_t N = T::shape().size();
    return [&]<auto... I>(std::index_sequence<I...>) {
        constexpr size_t M = N - 1;
        return [&]<auto... J>(std::index_sequence<J...>) {
            constexpr auto tuple = std::make_tuple(T::shape()[I]...);
            std::array<size_t, sizeof...(I)> offsets{};
            offsets[offsets.size() - 1] = index;
            return tens.template at<std::get<J>(tuple)...>(offsets);
        }(std::make_index_sequence<M>{});
    }(std::make_index_sequence<N>{});
}
// transpose tensor using index permutation
template <tensorial T, size_t... index_permutation>
    requires(fixed_tensor<T> && !scalar<T> && valid_index_permutation<index_permutation...>)
auto transpose_tensor(T &tens, std::index_sequence<index_permutation...> /*unused*/) {
    static_assert(T::shape().size() <= sizeof...(index_permutation), "invalid permutation");
    constexpr size_t N = sizeof...(index_permutation);
    using V = typename T::value_type;
    return [&]<auto... I>(std::index_sequence<I...>) {
        std::array<size_t, sizeof...(I)> strides{};
        strides.fill(1);
        std::array<size_t, sizeof...(I)> indices{index_permutation...};
        for (int i = 0; i < sizeof...(I); i++) {
            int perm_index = indices[i];
            if (perm_index < tens.strides().size()) {
                strides[i] = tens.strides()[perm_index];
            }
        }
        return tensor_ref<V, T::shape(index_permutation)...>(tens.data(), strides);
    }(std::make_index_sequence<N>{});
}
// overload to transpose by reversing shape of tensor
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto transpose_tensor(T &tens) {
    constexpr size_t N = T::shape().size();
    using V = typename T::value_type;
    return [&]<auto... I>(std::index_sequence<I...>) {
        auto strides = tens.strides();
        std::reverse(strides.begin(), strides.end());
        constexpr auto shape_tuple = reverse<static_cast<int>(T::shape()[I])...>();
        return tensor_ref<V, std::get<I>(shape_tuple)...>(tens.data(), strides);
    }(std::make_index_sequence<N>{});
}
// const overload to transpose tensor with index permutation
template <tensorial T, size_t... index_permutation>
    requires(fixed_tensor<T> && !scalar<T> && valid_index_permutation<index_permutation...>)
auto transpose_tensor(const T &tens, std::index_sequence<index_permutation...> /*unused*/) {
    static_assert(T::shape().size() <= sizeof...(index_permutation), "invalid permutation");
    constexpr size_t N = sizeof...(index_permutation);
    using V = typename T::value_type;
    return [&]<auto... I>(std::index_sequence<I...>) {
        std::array<size_t, sizeof...(I)> strides{};
        strides.fill(1);
        std::array<size_t, sizeof...(I)> indices{index_permutation...};
        for (int i = 0; i < sizeof...(I); i++) {
            int perm_index = indices[i];
            if (perm_index < tens.strides().size()) {
                strides[i] = tens.strides()[perm_index];
            }
        }
        return tensor_ref<const V, T::shape(index_permutation)...>(tens.data(), strides);
    }(std::make_index_sequence<N>{});
}
// const overload to transpose by reversing shape of tensor
template <tensorial T>
    requires(fixed_tensor<T> && !scalar<T>)
auto transpose_tensor(const T &tens) {
    constexpr size_t N = T::shape().size();
    using V = typename T::value_type;
    return [&]<auto... I>(std::index_sequence<I...>) {
        auto strides = tens.strides();
        std::reverse(strides.begin(), strides.end());
        constexpr auto shape_tuple = reverse<static_cast<int>(T::shape()[I])...>();
        return tensor_ref<const V, std::get<I>(shape_tuple)...>(tens.data(), strides);
    }(std::make_index_sequence<N>{});
}
// runtime bounds check for tensors, indexes dimension not shape
template <int... sizes> bool bounds_check(std::integral auto... indices) {
    std::array<size_t, sizeof...(indices)> index_arr{static_cast<size_t>(indices)...};
    std::array<size_t, sizeof...(sizes)> size_arr{sizes...};
    size_t first_not_one = get_first_not_one<sizes...>();
    assert(sizeof...(indices) <= sizeof...(sizes) - first_not_one); // too many indices.
    for (int i = first_not_one; i < sizeof...(indices) + first_not_one; i++) {
        if (index_arr[i - first_not_one] >= size_arr[i]) {
            return false;
        }
    }
    return true;
}
// result types
template <typename T, typename U>
using mult_type = decltype(std::declval<typename T::value_type &>() * std::declval<typename U::value_type &>());
template <typename T, typename U>
using div_type = decltype(std::declval<typename T::value_type &>() / std::declval<typename U::value_type &>());
template <typename T, typename U>
using sum_type = decltype(std::declval<typename T::value_type &>() + std::declval<typename U::value_type &>());
template <typename T, typename U>
using diff_type = decltype(std::declval<typename T::value_type &>() - std::declval<typename U::value_type &>());
template <typename T>
using squared_type = decltype(std::declval<typename T::value_type &>() * std::declval<typename T::value_type &>());
// get the underlying type of a generic tensor T
template <typename T> struct underlying_type {};
template <template <typename, int...> class U, typename T, int... sizes> struct underlying_type<U<T, sizes...>> {
    using type = T;
};
template <template <typename, int...> class U, quantitative T, int... sizes> struct underlying_type<U<T, sizes...>> {
    using type = typename T::value_type;
};
// ---------------------------------------------------------------------------------------------------------------

// Abstract base class for tensors and tensor refs, uses CRTP to get static
// polymorphism U is the derived tensor class, T is the element data type.
// dynamic shape can be created with the 'dynamic_shape' variable
template <class U, typename T, int... sizes> class tensor_base {
    static_assert(((sizes > 0) && ...), "All sizes must be strictly positive.");

  protected:
    tensor_base(){}; // since this is an abstract base class only derived classes can create a tensor_base

  public:
    using value_type = T; // element data type

    // assignment from another tensor of the same type (overrides derived class copy constructors)
    tensor_base &operator=(const tensor_base &other) {
        std::copy(other.cbegin(), other.cend(), this->begin());
        return *this;
    }
    // assignment from another tensor of the same shape
    template <tensor_shape<sizes...> M> U &operator=(const M &other) {
        using V = typename M::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        std::transform(other.begin(), other.end(), this->begin(), [](V x) { return static_cast<T>(x); });
        return static_cast<U &>(*this);
    }
    // Gets the shape of the block. This is size of each dimension
    static constexpr auto shape() {
        return std::array<size_t, sizeof...(sizes)>{static_cast<size_t>(sizes)...};
    }
    static constexpr auto squeezed_shape() { return squeeze_array<sizes...>(); }
    // Get the size of a specific dimension in the block's shape
    static constexpr size_t shape(size_t dim) {
        if (dim < shape().size()) {
            return shape().at(dim);
        }
        return 1;
    }
    // The number of dimensions the block has (not the number of sizes)
    static constexpr size_t order() {
        size_t order = 0;
        for (const auto &dim : shape()) {
            if (dim > 1) {
                order++;
            }
        }
        return order;
    }
    // total number of elements in the tensor
    static constexpr size_t size() {
        size_t product{1};
        for (int i = 0; i < shape().size(); i++) {
            product *= shape(i);
        }
        return product;
    }
    // A stride calculation used to compute the index into the flat array, this
    // overload uses a std::array as indices.
    static size_t compute_flat_index(const std::array<size_t, sizeof...(sizes)> &strides,
                                     const std::array<size_t, sizeof...(sizes)> &indices) {
        return std::inner_product(indices.begin(), indices.end(), strides.begin(), size_t(0));
    }
    // A stride calculation used to compute the index into the flat array, this
    // overload uses separate parameters for each index.
    static size_t compute_flat_index(const std::array<size_t, sizeof...(sizes)> &strides,
                                     std::integral auto... indices) {
        static_assert(sizeof...(indices) == sizeof...(sizes), "Number of indices must match number of sizes.");
        std::array<size_t, sizeof...(sizes)> indices_arr{static_cast<size_t>(indices)...};
        return std::inner_product(indices_arr.begin(), indices_arr.end(), strides.begin(), size_t(0));
    }
    // calculate strides from the shape of the tensor
    static constexpr std::array<size_t, sizeof...(sizes)> compute_strides() {
        std::array<size_t, sizeof...(sizes)> result{};
        size_t stride = 1;
        for (size_t i = 0; i < sizeof...(sizes); i++) {
            result[i] = stride;
            stride *= shape(i);
        }
        return result;
    }
    // Check that the sizes of a block will exactly fit in the tensor
    template <int... other_sizes> static constexpr bool check_blocks_fit() {
        static_assert(((other_sizes >= 0) && ...)); //  All sizes must be >= 0
        const std::array<size_t, sizeof...(other_sizes)> other_szs{static_cast<size_t>(other_sizes)...};
        for (size_t i = 0; i < sizeof...(other_sizes); i++) {
            if (shape(i) % other_szs.at(i) != 0) {
                return false;
            }
        }
        return true;
    }
    // iterators
    // -------------------------------------------------------------------------------------------------------
    // forward iterator for non-const element access
    struct tensor_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = value_type *;
        using reference = value_type &;
        tensor_iterator(pointer ptr, std::array<size_t, sizeof...(sizes)> strides)
            : _ptr(ptr), _index(0), _strides(strides) {}
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        tensor_iterator &operator++() {
            for (size_t i = 0; i < sizeof...(sizes); i++) {
                if (_indices[i] == shape(i) - 1) {
                    _indices[i] = 0;
                    if (i == _indices.size() - 1) {
                        _ptr++;
                        return *this;
                    }
                } else {
                    _indices[i]++;
                    break;
                }
            }
            size_t index_inc = compute_flat_index(_strides, _indices) - _index;
            _ptr += index_inc;
            _index += index_inc;
            return *this;
        }

        // Postfix increment
        tensor_iterator operator++(int) {
            tensor_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const tensor_iterator &a, const tensor_iterator &b) { return a._ptr == b._ptr; };
        friend bool operator!=(const tensor_iterator &a, const tensor_iterator &b) { return a._ptr != b._ptr; };

      private:
        pointer _ptr;
        size_t _index;
        std::array<size_t, sizeof...(sizes)> _indices{};
        std::array<size_t, sizeof...(sizes)> _strides;
    };
    // forward iterator for const element access
    struct const_tensor_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = value_type *;
        using reference = value_type &;
        const_tensor_iterator(pointer ptr, std::array<size_t, sizeof...(sizes)> strides)
            : _ptr(ptr), _index(0), _strides(strides) {}
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        const_tensor_iterator &operator++() {
            for (size_t i = 0; i < _indices.size(); i++) {
                if (_indices[i] == shape(i) - 1) {
                    _indices[i] = 0;
                    if (i == _indices.size() - 1) {
                        _ptr++;
                        return *this;
                    }
                } else {
                    _indices[i]++;
                    break;
                }
            }
            size_t index_inc = compute_flat_index(_strides, _indices) - _index;
            _ptr += index_inc;
            _index += index_inc;
            return *this;
        }

        // Postfix increment
        const_tensor_iterator operator++(int) {
            const_tensor_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_tensor_iterator &a, const const_tensor_iterator &b) {
            return a._ptr == b._ptr;
        };
        friend bool operator!=(const const_tensor_iterator &a, const const_tensor_iterator &b) {
            return a._ptr != b._ptr;
        };

      private:
        pointer _ptr;
        size_t _index;
        std::array<size_t, sizeof...(sizes)> _indices{};
        std::array<size_t, sizeof...(sizes)> _strides;
    };
    // the strides between elements for each dimension of the tensor
    std::array<size_t, sizeof...(sizes)> strides() const { return static_cast<const U *>(this)->strides(); }
    // pointer to start of tensor data (data may not be contiguous if this is a tensor_ref but otherwise will be)
    T *data() { return static_cast<U *>(this)->data(); }
    // const overload of data pointer
    const T *data() const { return static_cast<const U *>(this)->data(); }
    // get a block of a tensor at an index offset for each dimension
    template <int... ref_sizes> tensor_ref<T, ref_sizes...> at(std::integral auto... indices) {
        size_t start = compute_flat_index(strides(), indices...);
        std::array<size_t, sizeof...(ref_sizes)> ref_strides;
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + sizeof...(ref_sizes), ref_strides.begin());
        return tensor_ref<T, ref_sizes...>(&data()[start], ref_strides);
    }
    // overload for array of indices
    template <int... ref_sizes> tensor_ref<T, ref_sizes...> at(std::array<size_t, sizeof...(sizes)> indices) {
        size_t start = tensor_base<tensor_ref<T, sizes...>, T, sizes...>::compute_flat_index(strides(), indices);
        std::array<size_t, sizeof...(ref_sizes)> ref_strides;
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + sizeof...(ref_sizes), ref_strides.begin());
        return tensor_ref<T, ref_sizes...>(&data()[start], ref_strides);
    }
    // const overload
    template <int... ref_sizes> tensor_ref<const T, ref_sizes...> at(std::integral auto... indices) const {
        size_t start = tensor_base<tensor_ref<T, sizes...>, T, sizes...>::compute_flat_index(strides(), indices...);
        std::array<size_t, sizeof...(ref_sizes)> ref_strides;
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + sizeof...(ref_sizes), ref_strides.begin());
        return tensor_ref<const T, ref_sizes...>(&data()[start], ref_strides);
    }
    // const overload for array of indices
    template <int... ref_sizes>
    tensor_ref<const T, ref_sizes...> at(std::array<size_t, sizeof...(sizes)> indices) const {
        size_t start = tensor_base<tensor_ref<T, sizes...>, T, sizes...>::compute_flat_index(strides(), indices);
        std::array<size_t, sizeof...(ref_sizes)> ref_strides;
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + sizeof...(ref_sizes), ref_strides.begin());
        return tensor_ref<const T, ref_sizes...>(&data()[start], ref_strides);
    }
    // rows
    auto rows() {
        static_assert(U::shape().size() <= 2, "tensor has too many dimensions.");
        constexpr auto t = squint::remove_trailing<1, U::shape(1)>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            return this->block<std::get<I>(t)...>();
        }(std::make_index_sequence<N>{});
    }
    // cols
    auto cols() {
        static_assert(U::shape().size() <= 2, "tensor has too many dimensions.");
        constexpr auto t = squint::remove_trailing<U::shape(0)>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            return this->block<std::get<I>(t)...>();
        }(std::make_index_sequence<N>{});
    }
    // rows const
    auto rows() const {
        static_assert(U::shape().size() <= 2, "tensor has too many dimensions.");
        constexpr auto t = squint::remove_trailing<1, U::shape(1)>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            return this->block<std::get<I>(t)...>();
        }(std::make_index_sequence<N>{});
    }
    // cols const
    auto cols() const {
        static_assert(U::shape().size() <= 2, "tensor has too many dimensions.");
        constexpr auto t = squint::remove_trailing<U::shape(0)>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            return this->block<std::get<I>(t)...>();
        }(std::make_index_sequence<N>{});
    }
    // get a tensor of equal shape tensors to iterator over
    template <int... block_sizes> auto block() {
        static_assert(check_blocks_fit<block_sizes...>(), "Blocks do not fit the tensor");
        constexpr size_t N = shape().size();
        return [&]<auto... I>(std::index_sequence<I...>) {
            tensor<tensor_ref<T, block_sizes...>, (shape(I) / tensor_ref<T, block_sizes...>::shape(I))...> blocks{};
            std::array<size_t, sizeof...(sizes)> block_sizes_arr{tensor_ref<T, block_sizes...>::shape(I)...};
            std::array<size_t, sizeof...(sizes)> indices{};
            std::array<size_t, sizeof...(sizes)> tensor_sizes_arr{sizes...};
            std::array<size_t, sizeof...(sizes)> block_counts{};
            for (int i = 0; i < sizeof...(sizes); i++) {
                block_counts[i] = tensor_sizes_arr[i] / block_sizes_arr[i];
            }

            for (auto &blk : blocks) {
                std::array<size_t, sizeof...(sizes)> tensor_indices;
                for (int i = 0; i < sizeof...(sizes); i++) {
                    tensor_indices[i] = indices[i] * block_sizes_arr[i];
                }
                auto ref = at<block_sizes...>(tensor_indices);
                blk.set_ref(ref);
                for (size_t i = 0; i < sizeof...(sizes); i++) {
                    if (indices[i] == block_counts[i] - 1) {
                        indices[i] = 0;
                    } else {
                        indices[i]++;
                        break;
                    }
                }
            }
            return blocks;
        }(std::make_index_sequence<N>{});
    }

    // const overload
    template <int... block_sizes> auto block() const {
        static_assert(check_blocks_fit<block_sizes...>(), "Blocks do not fit the tensor");
        constexpr size_t N = shape().size();
        return [&]<auto... I>(std::index_sequence<I...>) {
            tensor<tensor_ref<const T, block_sizes...>, (shape(I) / tensor<T, block_sizes...>::shape(I))...> blocks{};
            std::array<size_t, sizeof...(sizes)> block_sizes_arr{tensor_ref<T, block_sizes...>::shape(I)...};
            std::array<size_t, sizeof...(sizes)> indices{};
            std::array<size_t, sizeof...(sizes)> tensor_sizes_arr{sizes...};
            std::array<size_t, sizeof...(sizes)> block_counts{};
            for (int i = 0; i < sizeof...(sizes); i++) {
                block_counts[i] = tensor_sizes_arr[i] / block_sizes_arr[i];
            }

            for (auto &blk : blocks) {
                std::array<size_t, sizeof...(sizes)> tensor_indices;
                for (int i = 0; i < sizeof...(sizes); i++) {
                    tensor_indices[i] = indices[i] * block_sizes_arr[i];
                }
                auto ref = at<block_sizes...>(tensor_indices);
                blk.set_ref(ref);
                for (size_t i = 0; i < sizeof...(sizes); i++) {
                    if (indices[i] == block_counts[i] - 1) {
                        indices[i] = 0;
                    } else {
                        indices[i]++;
                        break;
                    }
                }
            }
            return blocks;
        }(std::make_index_sequence<N>{});
    }
    // Fortran style indexing (**not used**)
    // auto operator()(std::integral auto... indices) {
    //     static_assert(sizeof...(indices) <= sizeof...(sizes), "Too many indices for this tensor.");
    //     assert(bounds_check<sizes...>(indices...)); // Index out of bounds.
    //     return index_tensor(static_cast<U &>(*this), indices...);
    // }
    // const auto operator()(std::integral auto... indices) const {
    //     static_assert(sizeof...(indices) <= sizeof...(sizes), "Too many indices for this tensor.");
    //     assert(bounds_check<sizes...>(indices...)); // Index out of bounds.
    //     return index_tensor(static_cast<const U &>(*this), indices...);
    // }
    // C style indexing. Block is created where first dimension != 1 is set to 1 and index of first dimension != 1 is
    // set to index. Result is simplified so all leading 1s are removed.
    auto operator[](size_t index) {
        static_assert(tensor_base<tensor_ref<T, sizes...>, T, sizes...>::size() > 1, "Cannot index tensor of size 1.");
        assert(bounds_check<sizes...>(index)); // Index out of bounds.
        return index_tensor(static_cast<U &>(*this), index);
    }
    // const overload for C style indexing
    auto operator[](size_t index) const {
        static_assert(tensor_base<tensor_ref<T, sizes...>, T, sizes...>::size() > 1, "Cannot index tensor of size 1.");
        assert(bounds_check<sizes...>(index)); // Index out of bounds.
        return index_tensor(static_cast<const U &>(*this), index);
    }
    // transpose shape using an index permutation
    template <int... index_permutation> auto transpose() {
        return transpose_tensor(static_cast<U &>(*this), std::index_sequence<index_permutation...>()).remove_trailing();
    }
    // specialize empty parameter list of transpose to be matrix transpose
    auto transpose() { return transpose<1, 0>(); }
    // const overload
    template <int... index_permutation> auto transpose() const {
        return transpose_tensor(static_cast<const U &>(*this), std::index_sequence<index_permutation...>())
            .remove_trailing();
    }
    // const overload
    auto transpose() const { return transpose<1, 0>(); }
    void apply(T (*F)(T)) {
        for (auto &elem : *this) {
            elem = F(elem);
        }
    }
    auto remove_trailing() {
        constexpr auto t = squint::remove_trailing<sizes...>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            std::array<size_t, N> new_strides{};
            for (int i = 0; i < N; i++) {
                new_strides[i] = strides()[i];
            }
            return tensor_ref<T, std::get<I>(t)...>(data(), new_strides);
        }(std::make_index_sequence<N>{});
    }
    // remove all trailing and leading 1s from the shape
    auto simplify_shape() {
        constexpr auto t = trim<sizes...>();
        constexpr size_t offset = get_first_not_one<sizes...>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            std::array<size_t, N> simple_strides{};
            for (int i = offset; i < N + offset; i++) {
                simple_strides[i - offset] = strides()[i];
            }
            return tensor_ref<T, std::get<I>(t)...>(data(), simple_strides);
        }(std::make_index_sequence<N>{});
    }
    auto remove_trailing() const {
        constexpr auto t = squint::remove_trailing<sizes...>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            std::array<size_t, N> new_strides{};
            for (int i = 0; i < N; i++) {
                new_strides[i] = strides()[i];
            }
            return tensor_ref<const T, std::get<I>(t)...>(data(), new_strides);
        }(std::make_index_sequence<N>{});
    }
    // const overload
    auto simplify_shape() const {
        constexpr auto t = trim<sizes...>();
        constexpr size_t offset = get_first_not_one<sizes...>();
        constexpr size_t N = std::tuple_size<decltype(t)>::value;
        return [&]<auto... I>(std::index_sequence<I...>) {
            std::array<size_t, N> simple_strides{};
            for (int i = offset; i < sizeof...(sizes); i++) {
                simple_strides[i - offset] = strides()[i];
            }
            return tensor_ref<const T, std::get<I>(t)...>(data(), simple_strides);
        }(std::make_index_sequence<N>{});
    }
    auto as_ref() { return tensor_ref<T, sizes...>(data(), strides()); }
    auto as_ref() const { return tensor_ref<const T, sizes...>(data(), strides()); }
    // remove the last dimension not equal to 1 and index the tensor at 'index' for last dimension not equal to 1
    auto demote_tensor(size_t index) { return demote_fixed_tensor(index, static_cast<U &>(*this)); }
    // const overload
    auto demote_tensor(size_t index) const { return demote_fixed_tensor(index, static_cast<const U &>(*this)); }
    // produce a deep copy of the tensor
    tensor<typename std::remove_const<T>::type, sizes...> copy() const {
        tensor<typename std::remove_const<T>::type, sizes...> the_copy{};
        std::transform(begin(), end(), the_copy.begin(),
                       [](const T &a) -> typename std::remove_const<T>::type { return a; });
        return the_copy;
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor<Q, sizes...> copy_as() const {
        tensor<Q, sizes...> the_copy{};
        std::transform(begin(), end(), the_copy.begin(), [](const T &a) -> Q { return static_cast<const Q>(a); });
        return the_copy;
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor_ref<Q, sizes...> view_as() {
        return tensor_ref<Q, sizes...>(reinterpret_cast<Q *>(data()), strides());
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor_ref<Q, sizes...> view_as() const {
        return tensor_ref<Q, sizes...>(reinterpret_cast<const Q *>(data()), strides());
    }
    // non-const iterator begin. will iterate in column major order
    tensor_iterator begin() { return tensor_iterator(data(), strides()); }
    // non-const iterator end. will iterate in column major order
    tensor_iterator end() {
        const std::array<size_t, sizeof...(sizes)> indices{static_cast<size_t>(sizes - 1)...};
        T *last_ptr = &data()[compute_flat_index(strides(), indices)];
        return tensor_iterator(++last_ptr, strides());
    }
    // const iterator begin. will iterate in column major order
    const_tensor_iterator begin() const { return const_tensor_iterator(data(), strides()); }
    const_tensor_iterator cbegin() const { return begin(); }
    // const iterator end. will iterate in column major order
    const_tensor_iterator end() const {
        const std::array<size_t, sizeof...(sizes)> indices{static_cast<size_t>(sizes - 1)...};
        const T *last_ptr = &data()[compute_flat_index(strides(), indices)];
        return const_tensor_iterator(++last_ptr, strides());
    }
    const_tensor_iterator cend() const { return end(); }

    // unary minus operator to negate a copy of this tensor
    tensor<T, sizes...> operator-() const {
        auto result = copy();
        for (auto &elem : result) {
            elem = -elem;
        }
        return result;
    }
    // scale
    tensor_base &operator*=(const T &s) {
        // in case s is a scalar ref of this tensor, we need to copy to a new variable
        value_type s_copy = s;
        for (auto &elem : *this) {
            elem *= s_copy;
        }
        return *this;
    }
    tensor_base &operator/=(const T &s) {
        // in case s is a scalar ref of this tensor, we need to copy to a new variable
        value_type s_copy = s;
        for (auto &elem : *this) {
            elem /= s_copy;
        }
        return *this;
    }
    // addition and subtraction
    tensor_base &operator+=(tensor_shape<sizes...> auto const &rhs) {
        auto it2 = rhs.begin();
        for (auto it1 = begin(); it2 != rhs.end(); it1++, it2++) {
            *it1 += *it2;
        }
        return *this;
    }
    tensor_base &operator-=(tensor_shape<sizes...> auto const &rhs) {
        auto it2 = rhs.begin();
        for (auto it1 = begin(); it2 != rhs.end(); it1++, it2++) {
            *it1 -= *it2;
        }
        return *this;
    }
    tensor<T, sizes...> operator+(const tensor_base &other) const {
        tensor<T, sizes...> result(*this);
        result += other;
        return result;
    }
    tensor<T, sizes...> operator-(const tensor_base &other) const {
        tensor<T, sizes...> result(*this);
        result -= other;
        return result;
    }
    // insert flat
    template <fixed_tensor M> void insert_flat(const M &other) {
        using V = typename M::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        static_assert(tensor_base<tensor_ref<T, sizes...>, T, sizes...>::size() == other.size(),
                      "tensors must have the same number of elements.");
        std::transform(other.begin(), other.end(), this->begin(), [](V x) { return static_cast<T>(x); });
    }
    static tensor<T, sizes...> I() {
        static_assert(order() == 2, "must be order 2");
        static_assert(shape(0) == shape(1), "must be square");
        auto eye = tensor<T, sizes...>();
        for (size_t i = 0; i < eye.shape(0); i++) {
            eye[i][i] = 1;
        }
        return eye;
    }
    static tensor<T, sizes...> ones() { return tensor<T, sizes...>(1); }
    static tensor<T, sizes...> zeros() { return tensor<T, sizes...>(0); }
    static tensor<T, sizes...> fill(const T &value) { return tensor<T, sizes...>(value); }
    static tensor<T, sizes...> diag(const tensor<T, shape(0)> &diag) {
        static_assert(order() == 2, "must be order 2");
        static_assert(shape(0) == shape(1), "must be square");
        auto result = tensor<T, sizes...>();
        for (size_t i = 0; i < diag.size(); i++) {
            result[i][i] = diag[i];
        }
        return result;
    }
    static tensor<T, sizes...> diag(const T &diag) {
        static_assert(order() == 2, "must be order 2");
        static_assert(shape(0) == shape(1), "must be square");
        auto result = tensor<T, sizes...>();
        for (size_t i = 0; i < result.shape(0); i++) {
            result[i][i] = diag;
        }
        return result;
    }
    static tensor<T, sizes...> random(T min = 0, T max = 1) {
        static_assert(quantitative<T> || std::is_arithmetic<T>::value, "must be arithmetic or quantitative type");
        auto result = tensor<T, sizes...>();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        for (auto &elem : result) {
            elem = dis(gen);
        }
        return result;
    }
};

// A reference to a tensor or another tensor_ref
template <typename T, int... sizes> class tensor_ref : public tensor_base<tensor_ref<T, sizes...>, T, sizes...> {
  public:
    using value_type = T; // data type of the elements
    tensor_ref() : _data(nullptr), _strides({}) {}
    tensor_ref(T *data, const std::array<size_t, sizeof...(sizes)> &strides) : _data(data), _strides(strides) {}
    tensor_ref &operator=(const tensor_ref &other) { return tensor_base<tensor_ref, T, sizes...>::operator=(other); }
    template <tensor_shape<sizes...> M> tensor_ref<T, sizes...> &operator=(const M &other) {
        return tensor_base<tensor_ref, T, sizes...>::operator=(other);
    }
    std::array<size_t, sizeof...(sizes)> strides() const { return _strides; }
    T *data() { return _data; }
    const T *data() const { return _data; }
    void set_ref(tensor_ref &ref) {
        _data = ref.data();
        _strides = ref.strides();
    }

  private:
    T *_data;                                      // pointer to start of data being referenced
    std::array<size_t, sizeof...(sizes)> _strides; // strides for each dimension in the array
};

// Main tensor class
template <typename T, int... sizes> class tensor : public tensor_base<tensor<T, sizes...>, T, sizes...> {
    // // this is only in tensor class, tensor refs can end in 1 when they reference a higher order tensor
    // static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::shape(sizeof...(sizes) - 1) != 1,
    //               "Shape cannot end with 1. This is always implicit.");

  public:
    using value_type = T; // data type of the elements
    tensor() = default;
    // Construct a new tensor filled with a constant value of the element type
    tensor(const T &value) { _data.fill(value); }
    // Construct a new tensor filled with repeating blocks of the given tensor
    template <template <typename, int...> class U, typename V, int... other_sizes>
        requires fixed_tensor<U<V, other_sizes...>>
    tensor(const U<V, other_sizes...> &other) {
        static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::size() %
                              tensor_base<tensor<T, sizes...>, T, other_sizes...>::size() ==
                          0,
                      "Incompatible tensor sizes.");
        static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::template check_blocks_fit<other_sizes...>(),
                      "Incompatible tensor shapes.");
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        if (sizeof...(other_sizes) == 0) {
            // scalar
            _data.fill(other.data()[0]);
            return;
        }
        const int count = tensor_base<tensor<T, sizes...>, T, sizes...>::size() /
                          tensor_base<tensor<T, sizes...>, T, other_sizes...>::size();
        std::array<size_t, sizeof...(sizes)> offsets{0};
        for (int i = 0; i < count; i++) {
            this->template at<other_sizes...>(offsets) = other;
            for (int i = 0; i < sizeof...(sizes); i++) {
                if (i < other.shape().size()) {
                    offsets[i] += other.shape(i);
                } else {
                    offsets[i]++;
                }
                if (offsets[i] == this->shape(i)) {
                    offsets[i] = 0;
                } else {
                    break;
                }
            }
        }
    }
    // Construct a new tensor from a blocks of tensors or tensor_refs, blocks are stored in column major order
    template <template <typename, int...> class U, typename V, int... other_sizes>
        requires fixed_tensor<U<V, other_sizes...>>
    tensor(std::initializer_list<U<V, other_sizes...>> blocks) {
        static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::template check_blocks_fit<other_sizes...>(),
                      "Blocks do not fit the tensor.");
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        std::array<size_t, sizeof...(other_sizes)> other{static_cast<size_t>(other_sizes)...};
        std::array<size_t, sizeof...(sizes)> offsets{0};
        for (const auto &blk : blocks) {
            tensor_ref<T, other_sizes...> ref = this->template at<other_sizes...>(offsets);
            ref = blk;
            for (int i = 0; i < sizeof...(sizes); i++) {
                if (i < other.size()) {
                    offsets[i] += blk.shape(i);
                } else {
                    offsets[i]++;
                }
                if (offsets[i] == this->shape(i)) {
                    offsets[i] = 0;
                } else {
                    break;
                }
            }
        }
    }
    // Construct a new tensor from a list of values in column major order
    template <scalar V> tensor(std::initializer_list<V> values) {
        int i = 0;
        for (const auto &value : values) {
            _data[i] = value;
            i++;
        }
    }
    tensor &operator=(const tensor &other) { return tensor_base<tensor, T, sizes...>::operator=(other); }
    template <tensor_shape<sizes...> M> tensor<T, sizes...> &operator=(const M &other) {
        return tensor_base<tensor, T, sizes...>::operator=(other);
    }
    T *data() { return _data.data(); }
    const T *data() const { return _data.data(); }
    std::array<size_t, sizeof...(sizes)> strides() const {
        return tensor_base<tensor<T, sizes...>, T, sizes...>::compute_strides();
    }
    // main tensors can be reshaped, not tensor_refs
    template <int... new_shape> tensor_ref<T, new_shape...> as_shape() {
        static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::size() ==
                          tensor_base<tensor<T, sizes...>, T, new_shape...>::size(),
                      "tensors must have the same number of elements.");
        return tensor_ref<T, new_shape...>(data(),
                                           tensor_base<tensor<T, sizes...>, T, new_shape...>::compute_strides());
    }
    template <int... new_shape> tensor_ref<const T, new_shape...> as_shape() const {
        static_assert(tensor_base<tensor<T, sizes...>, T, sizes...>::size() ==
                          tensor_base<tensor<T, sizes...>, T, new_shape...>::size(),
                      "tensors must have the same number of elements.");
        return tensor_ref<const T, new_shape...>(data(),
                                                 tensor_base<tensor<T, sizes...>, T, new_shape...>::compute_strides());
    }
    // flatten shape to a column vector
    tensor_ref<T, tensor_base<tensor<T, sizes...>, T, sizes...>::size()> flatten() {
        return tensor_ref<T, tensor_base<tensor<T, sizes...>, T, sizes...>::size()>(data(), {1});
    }
    tensor_ref<const T, tensor_base<tensor<T, sizes...>, T, sizes...>::size()> flatten() const {
        return tensor_ref<const T, tensor_base<tensor<T, sizes...>, T, sizes...>::size()>(data(), {1});
    }

  private:
    std::array<T, tensor_base<tensor<T, sizes...>, T, sizes...>::size()> _data{};
};

// Partial specialization for scalars
template <typename U, typename T> class tensor_base<U, T> {
  protected:
    tensor_base(){}; // only base classes can create a tensor_base
  public:
    using value_type = T; // data type of the elements
    // scalar shape is a zero size array
    static constexpr std::array<size_t, 0> shape() { return {}; }
    static constexpr auto squeezed_shape() { return shape(); }
    // Shape of any dimension is always 1
    static constexpr size_t shape(const size_t dim) { return 1; }
    // A scalar is a zeroth order tensor
    static constexpr size_t order() { return 0; }
    // Total number of elements is always 1
    static constexpr size_t size() { return 1; }
    std::array<size_t, 0> strides() const { return {}; }
    T *data() { return static_cast<U *>(this)->data(); }
    const T *data() const { return static_cast<const U *>(this)->data(); }
    // demote tensor is overloaded for scalars. Used to print
    auto demote_tensor(size_t index) { return tensor(*data()); }
    auto demote_tensor(size_t index) const { return tensor(*data()); }
    auto simplify_shape() { return tensor(*data()); }
    auto simplify_shape() const { return tensor(*data()); }
    // // overload reference operator of scalars to return pointer to data
    // T *operator&() { return data(); }
    // const T *operator&() const { return data(); }
    // 3-way comparison operator
    auto operator<=>(const tensor_base &rhs) const { return data()[0] <=> rhs.data()[0]; }
    // math operators
    tensor<T> operator-() const { return tensor<T>(-data()[0]); }
    tensor_base &operator+=(const T &rhs) {
        data()[0] += rhs;
        return *this;
    }
    tensor_base &operator-=(const T &rhs) {
        data()[0] -= rhs;
        return *this;
    }
    tensor_base &operator*=(const T &rhs) {
        data()[0] *= rhs;
        return *this;
    }
    tensor_base &operator/=(const T &rhs) {
        data()[0] /= rhs;
        return *this;
    }
    template <scalar S>
        requires(!tensorial<S>)
    tensor<T> operator+(const S &rhs) const {
        return tensor<T>(data()[0] + rhs);
    }
    template <scalar S>
        requires(!tensorial<S>)
    tensor<T> operator-(const S &rhs) const {
        return tensor<T>(data()[0] - rhs);
    }
    template <scalar S>
        requires(!tensorial<S>)
    tensor<T> operator*(const S &rhs) const {
        return tensor<T>(data()[0] * rhs);
    }
    template <scalar S>
        requires(!tensorial<S>)
    tensor<T> operator/(const S &rhs) const {
        return tensor<T>(data()[0] / rhs);
    }
    template <tensorial Q>
        requires(scalar<Q>)
    auto operator*(const Q &rhs) const {
        auto result = data()[0] * rhs.data()[0];
        return tensor<decltype(result)>(result);
    }
    template <tensorial Q>
        requires(scalar<Q>)
    auto operator/(const Q &rhs) const {
        auto result = data()[0] / rhs.data()[0];
        return tensor<decltype(result)>(result);
    }
    template <tensorial Q>
        requires(scalar<Q>)
    auto operator+(const Q &rhs) const {
        auto result = data()[0] + rhs.data()[0];
        return tensor<decltype(result)>(result);
    }
    template <tensorial Q>
        requires(scalar<Q>)
    auto operator-(const Q &rhs) const {
        auto result = data()[0] - rhs.data()[0];
        return tensor<decltype(result)>(result);
    }
    // below code will make API consistent so that templates will compile
    struct scalar_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = value_type *;
        using reference = value_type &;
        scalar_iterator(pointer ptr) : _ptr(ptr) {}
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        scalar_iterator &operator++() {
            _ptr += 1;
            return *this;
        }

        // Postfix increment
        scalar_iterator operator++(int) {
            scalar_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const scalar_iterator &a, const scalar_iterator &b) { return a._ptr == b._ptr; };
        friend bool operator!=(const scalar_iterator &a, const scalar_iterator &b) { return a._ptr != b._ptr; };

      private:
        pointer _ptr;
    };
    struct const_scalar_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = value_type *;
        using reference = value_type &;
        const_scalar_iterator(pointer ptr) : _ptr(ptr) {}
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        const_scalar_iterator &operator++() {
            _ptr += 1;
            return *this;
        }

        // Postfix increment
        const_scalar_iterator operator++(int) {
            const_scalar_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_scalar_iterator &a, const const_scalar_iterator &b) {
            return a._ptr == b._ptr;
        };
        friend bool operator!=(const const_scalar_iterator &a, const const_scalar_iterator &b) {
            return a._ptr != b._ptr;
        };

      private:
        pointer _ptr;
    };
    scalar_iterator begin() { return scalar_iterator(data()); }
    scalar_iterator end() { return scalar_iterator(data() + 1); }
    const_scalar_iterator begin() const { return const_scalar_iterator(data()); }
    const_scalar_iterator cbegin() const { return begin(); }
    const_scalar_iterator end() const { return const_scalar_iterator(data() + 1); }
    const_scalar_iterator cend() const { return end(); }
    auto operator[](size_t index)
        requires(!tensorial<T>)
    {
        assert(index == 0); // Index out of bounds.
        return tensor_ref<T>(data());
    }
    auto operator[](size_t index) const
        requires(!tensorial<T>)
    {
        assert(index == 0); // Index out of bounds.
        return tensor_ref<const T>(data());
    }
    auto transpose() { return tensor_ref<T>(data()); }
    auto transpose() const { return tensor_ref<const T>(data()); }
    auto rows() {
        auto ref = tensor_ref<T>(data());
        return tensor_ref<tensor_ref<T>>(ref);
    }
    // cols
    auto cols() {
        auto ref = tensor_ref<T>(data());
        return tensor_ref<tensor_ref<T>>(ref);
    }
    // rows const
    auto rows() const {
        const auto ref = tensor_ref<const T>(data());
        return tensor_ref<const tensor_ref<const T>>(ref);
    }
    // cols const
    auto cols() const {
        const auto ref = tensor_ref<const T>(data());
        return tensor_ref<const tensor_ref<const T>>(ref);
    }
    // produce a deep copy of the tensor
    tensor<typename std::remove_const<T>::type> copy() const {
        tensor<typename std::remove_const<T>::type> the_copy{};
        std::transform(begin(), end(), the_copy.begin(),
                       [](const T &a) -> typename std::remove_const<T>::type { return a; });
        return the_copy;
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor<Q> copy_as() const {
        tensor<Q> the_copy{};
        std::transform(begin(), end(), the_copy.begin(), [](const T &a) -> Q { return static_cast<const Q>(a); });
        return the_copy;
    }
};

template <typename T> class tensor_ref<T> : public tensor_base<tensor_ref<T>, T> {
  public:
    using value_type = T; // data type of the elements
    tensor_ref() : _data(nullptr) {}
    // tensor_ref scalar stores pointer to single element
    tensor_ref(T *data, const std::array<size_t, 0> &strides) : _data(data) {}
    tensor_ref(T *data) : _data(data) {}
    tensor_ref(scalar auto &other) { _data = &other; }
    //  implicit conversion to T&
    operator const T &() const { return _data[0]; }
    operator T &() { return _data[0]; }
    tensor_ref &operator=(scalar auto const &other) {
        _data[0] = other;
        return *this;
    }
    // assignment should copy data, not the pointer
    tensor_ref &operator=(const tensor_ref &other) {
        _data[0] = other;
        return *this;
    }
    T *data() { return _data; }
    const T *data() const { return _data; }
    std::array<size_t, 0> strides() const { return std::array<size_t, 0>{}; }
    void set_ref(tensor_ref &ref) { _data = ref.data(); }

  private:
    T *_data;
};

template <typename T> class tensor<T> : public tensor_base<tensor<T>, T> {
  public:
    using value_type = T; // data type of the elements
    tensor() = default;
    tensor(scalar auto const &other) : _elem(other) {}
    tensor(const T &other) : _elem(other) {}
    // implicit conversion to T
    operator T() const { return _elem; }
    tensor &operator=(scalar auto const &other) {
        _elem = other;
        return *this;
    }
    T *data() { return &_elem; }
    const T *data() const { return &_elem; }
    std::array<size_t, 0> strides() const { return std::array<size_t, 0>{}; }

  private:
    T _elem{};
};

// using definitions for common tensors
// doubles
using dscalar = tensor<double>;
using dmat2 = tensor<double, 2, 2>;
using dmat3 = tensor<double, 3, 3>;
using dmat4 = tensor<double, 4, 4>;
template <int m, int n> using dmat = tensor<double, m, n>;
using dvec2 = tensor<double, 2>;
using dvec3 = tensor<double, 3>;
using dvec4 = tensor<double, 4>;
template <int n> using dvec = tensor<double, n>;
using drvec2 = tensor<double, 1, 2>;
using drvec3 = tensor<double, 1, 3>;
using drvec4 = tensor<double, 1, 4>;
template <int n> using drvec = tensor<double, 1, n>;
template <int... sizes> using dtensor = tensor<double, sizes...>;
// floats
using fscalar = tensor<float>;
using fmat2 = tensor<float, 2, 2>;
using fmat3 = tensor<float, 3, 3>;
using fmat4 = tensor<float, 4, 4>;
template <int m, int n> using fmat = tensor<float, m, n>;
using fvec2 = tensor<float, 2>;
using fvec3 = tensor<float, 3>;
using fvec4 = tensor<float, 4>;
template <int n> using fvec = tensor<float, n>;
using frvec2 = tensor<float, 1, 2>;
using frvec3 = tensor<float, 1, 3>;
using frvec4 = tensor<float, 1, 4>;
template <int n> using frvec = tensor<float, 1, n>;
template <int... sizes> using ftensor = tensor<float, sizes...>;

// Dynamic Tensors
// -----------------------------------------------------------------------------------------------------
template <typename U, typename T> class tensor_base<U, T, dynamic_shape> {
  protected:
    tensor_base(std::vector<size_t> shape)
        : _shape(shape){}; // since this is an abstract base class only derived
                           // classes can create a tensor_base

  public:
    using value_type = T; // element data type

    tensor_base &operator=(const tensor_base &other) {
        assert(shape() == other.shape()); // tensors must have the same shape
        std::copy(other.cbegin(), other.cend(), this->begin());
        return *this;
    }

    template <dynamic_tensor M> U &operator=(const M &other) {
        assert(shape() == other.shape()); // tensors must have the same shape
        using V = typename M::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        std::transform(other.begin(), other.end(), this->begin(), [](V x) { return static_cast<T>(x); });
        return static_cast<U &>(*this);
    }

    // Gets the shape of the block. This is size of each dimension
    const std::vector<size_t> &shape() const { return _shape; }
    std::vector<size_t> squeezed_shape() const { 
        std::vector<size_t> result{};
        for (const auto &dim : shape()) {
            if (dim > 1) {
                result.push_back(dim);
            }
        }
        return result;
    }
    // Get the size of a specific dimension in the block's shape
    size_t shape(const size_t dim) const {
        if (dim < shape().size()) {
            return shape().at(dim);
        }
        return 1;
    }

    // The number of dimensions the block has (not the number of sizes)
    size_t order() const {
        size_t order = 0;
        for (const auto &dim : shape()) {
            if (dim > 1) {
                order++;
            }
        }
        return order;
    }
    // total number of elements in the tensor
    size_t size() const {
        if (shape().size() == 0) {
            return 0;
        }
        size_t product{1};
        for (int i = 0; i < shape().size(); i++) {
            product *= shape().at(i);
        }
        return product;
    }
    // A stride calculation used to compute the index into the flat array, this
    // overload uses a std::array as indices.
    static size_t compute_flat_index(const std::vector<size_t> &strides, const std::vector<size_t> &indices) {
        assert(strides.size() == indices.size());
        return std::inner_product(indices.begin(), indices.end(), strides.begin(), size_t(0));
    }
    static size_t compute_flat_index(const std::vector<size_t> &strides, std::integral auto... indices) {
        std::vector<size_t> indices_vec{indices...};
        assert(strides.size() == indices_vec.size());
        return std::inner_product(indices_vec.begin(), indices_vec.end(), strides.begin(), size_t(0));
    }
    // calculate strides from the shape of the tensor
    std::vector<size_t> compute_strides() const {
        std::vector<size_t> result{};
        result.resize(shape().size());
        size_t stride = 1;
        for (size_t i = 0; i < shape().size(); i++) {
            result[i] = stride;
            stride *= shape().at(i);
        }
        return result;
    }
    // Check that the sizes of a block will exactly fit in the tensor
    bool check_blocks_fit(const std::vector<size_t> &other_shape) const {
        for (size_t i = 0; i < other_shape.size(); i++) {
            if (shape(i) % other_shape.at(i) != 0) {
                return false;
            }
        }
        return true;
    }
    // iterators
    // -------------------------------------------------------------------------------------------------------
    // forward iterator for non-const element access
    struct tensor_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = value_type *;
        using reference = value_type &;
        tensor_iterator(pointer ptr, const std::vector<size_t> &strides,
                        tensor_base<U, T, dynamic_shape> *tensor_base_ptr)
            : _ptr(ptr), _index(0), _strides(strides), _tensor_base_ptr(tensor_base_ptr) {
            _indices.resize(_tensor_base_ptr->shape().size(), 0);
        }
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        tensor_iterator &operator++() {
            for (size_t i = 0; i < _tensor_base_ptr->shape().size(); i++) {
                if (_indices[i] == _tensor_base_ptr->shape().at(i) - 1) {
                    _indices[i] = 0;
                    if (i == _indices.size() - 1) {
                        _ptr++;
                        return *this;
                    }
                } else {
                    _indices[i]++;
                    break;
                }
            }
            size_t index_inc = compute_flat_index(_strides, _indices) - _index;
            _ptr += index_inc;
            _index += index_inc;
            return *this;
        }

        // Postfix increment
        tensor_iterator operator++(int) {
            tensor_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const tensor_iterator &a, const tensor_iterator &b) { return a._ptr == b._ptr; };
        friend bool operator!=(const tensor_iterator &a, const tensor_iterator &b) { return a._ptr != b._ptr; };

      private:
        pointer _ptr;
        size_t _index;
        std::vector<size_t> _indices{};
        std::vector<size_t> _strides;
        tensor_base<U, T, dynamic_shape> *_tensor_base_ptr;
    };
    // forward iterator for const element access
    struct const_tensor_iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const T;
        using pointer = value_type *;
        using reference = value_type &;
        const_tensor_iterator(pointer ptr, const std::vector<size_t> &strides,
                              const tensor_base<U, T, dynamic_shape> *tensor_base_ptr)
            : _ptr(ptr), _index(0), _strides(strides), _tensor_base_ptr(tensor_base_ptr) {
            _indices.resize(_tensor_base_ptr->shape().size(), 0);
        }
        reference operator*() const { return *_ptr; }
        pointer operator->() { return _ptr; }

        // Prefix increment
        const_tensor_iterator &operator++() {
            for (size_t i = 0; i < _indices.size(); i++) {
                if (_indices[i] == _tensor_base_ptr->shape().at(i) - 1) {
                    _indices[i] = 0;
                    if (i == _indices.size() - 1) {
                        _ptr++;
                        return *this;
                    }
                } else {
                    _indices[i]++;
                    break;
                }
            }
            size_t index_inc = compute_flat_index(_strides, _indices) - _index;
            _ptr += index_inc;
            _index += index_inc;
            return *this;
        }

        // Postfix increment
        const_tensor_iterator operator++(int) {
            const_tensor_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_tensor_iterator &a, const const_tensor_iterator &b) {
            return a._ptr == b._ptr;
        };
        friend bool operator!=(const const_tensor_iterator &a, const const_tensor_iterator &b) {
            return a._ptr != b._ptr;
        };

      private:
        pointer _ptr;
        size_t _index;
        std::vector<size_t> _indices{};
        std::vector<size_t> _strides;
        const tensor_base<U, T, dynamic_shape> *_tensor_base_ptr;
    };

    std::vector<size_t> strides() const { return static_cast<const U *>(this)->strides(); }
    T *data() { return static_cast<U *>(this)->data(); }
    const T *data() const { return static_cast<const U *>(this)->data(); }

    tensor_ref<T, dynamic_shape> at(const std::vector<size_t> &ref_sizes, std::integral auto... indices) {
        size_t start = compute_flat_index(strides(), indices...);
        std::vector<size_t> ref_strides(ref_sizes.size(), 0);
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + ref_strides.size(), ref_strides.begin());
        return tensor_ref<T, dynamic_shape>(ref_sizes, &data()[start], ref_strides);
    }
    tensor_ref<T, dynamic_shape> at(const std::vector<size_t> &ref_sizes, const std::vector<size_t> &indices) {
        size_t start = compute_flat_index(strides(), indices);
        std::vector<size_t> ref_strides(ref_sizes.size(), 0);
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + ref_strides.size(), ref_strides.begin());
        return tensor_ref<T, dynamic_shape>(ref_sizes, &data()[start], ref_strides);
    }
    tensor_ref<const T, dynamic_shape> at(const std::vector<size_t> &ref_sizes, std::integral auto... indices) const {
        size_t start = compute_flat_index(strides(), indices...);
        std::vector<size_t> ref_strides(ref_sizes.size(), 0);
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + ref_strides.size(), ref_strides.begin());
        return tensor_ref<const T, dynamic_shape>(ref_sizes, &data()[start], ref_strides);
    }
    tensor_ref<const T, dynamic_shape> at(const std::vector<size_t> &ref_sizes,
                                          const std::vector<size_t> &indices) const {
        size_t start = compute_flat_index(strides(), indices);
        std::vector<size_t> ref_strides(ref_sizes.size(), 0);
        auto full_strides = strides();
        std::copy(full_strides.begin(), full_strides.begin() + ref_strides.size(), ref_strides.begin());
        return tensor_ref<const T, dynamic_shape>(ref_sizes, &data()[start], ref_strides);
    }
    // rows
    auto rows() {
        assert(shape().size() <= 2);
        std::vector<size_t> new_shape{1};
        if (shape(1) != 1) {
            new_shape.push_back(shape(1));
        }
        return this->block(new_shape);
    }
    // cols
    auto cols() {
        assert(shape().size() <= 2);
        return this->block({shape(0)});
    }
    // rows const
    auto rows() const {
        assert(shape().size() <= 2);
        std::vector<size_t> new_shape{1};
        if (shape(1) != 1) {
            new_shape.push_back(shape(1));
        }
        return this->block(new_shape);
    }
    // cols const
    auto cols() const {
        assert(shape().size() <= 2);
        return this->block({shape(0)});
    }
    auto block(const std::vector<size_t> &block_shape) {
        assert(check_blocks_fit(block_shape)); // Blocks do not fit the tensor
        auto block_sizes_arr = block_shape;
        while (block_sizes_arr.size() < shape().size()) {
            block_sizes_arr.push_back(1);
        }
        std::vector<size_t> indices{};
        indices.resize(shape().size());
        std::vector<size_t> block_counts{};
        for (int i = 0; i < shape().size(); i++) {
            block_counts.push_back(shape(i) / block_sizes_arr[i]);
        }
        tensor<tensor_ref<T, dynamic_shape>, dynamic_shape> blocks;
        blocks.reshape(block_counts);
        for (auto &blk : blocks) {
            std::vector<size_t> tensor_indices;
            tensor_indices.resize(shape().size());
            for (int i = 0; i < shape().size(); i++) {
                tensor_indices[i] = indices[i] * block_sizes_arr[i];
            }
            auto ref = at(block_shape, tensor_indices);
            blk.set_ref(ref);
            for (size_t i = 0; i < shape().size(); i++) {
                if (indices[i] == block_counts[i] - 1) {
                    indices[i] = 0;
                } else {
                    indices[i]++;
                    break;
                }
            }
        }
        return blocks;
    }
    auto block(const std::vector<size_t> &block_shape) const {
        assert(check_blocks_fit(block_shape)); // Blocks do not fit the tensor
        auto block_sizes_arr = block_shape;
        while (block_sizes_arr.size() < shape().size()) {
            block_sizes_arr.push_back(1);
        }
        std::vector<size_t> indices{};
        indices.resize(shape().size());
        std::vector<size_t> block_counts{};
        for (int i = 0; i < shape().size(); i++) {
            block_counts.push_back(shape(i) / block_sizes_arr[i]);
        }
        tensor<tensor_ref<const T, dynamic_shape>, dynamic_shape> blocks;
        blocks.reshape(block_counts);
        for (auto &blk : blocks) {
            std::vector<size_t> tensor_indices;
            tensor_indices.resize(shape().size());
            for (int i = 0; i < shape().size(); i++) {
                tensor_indices[i] = indices[i] * block_sizes_arr[i];
            }
            auto ref = at(block_shape, tensor_indices);
            blk.set_ref(ref);
            for (size_t i = 0; i < shape().size(); i++) {
                if (indices[i] == block_counts[i] - 1) {
                    indices[i] = 0;
                } else {
                    indices[i]++;
                    break;
                }
            }
        }
        return blocks;
    }
    auto operator[](size_t index) {
        size_t N = shape().size();
        if (size() == 1) {
            // auto s = strides();
            // size_t start = compute_flat_index(s, {index});
            return tensor_ref<T, dynamic_shape>({1}, data(), {1});
            // return at({1}, 0);
        }
        if (N == 1) {
            auto s = strides();
            size_t start = compute_flat_index(s, {index});
            return tensor_ref<T, dynamic_shape>({1}, &data()[start], {s[0]});
            // return at({1}, {index});
        }
        if (N == 2) {
            if (shape(0) == 1) {
                auto s = strides();
                size_t start = compute_flat_index(s, {0, index});
                return tensor_ref<T, dynamic_shape>({1}, &data()[start], {s[0]});
                // return at({1}, {0, index});
            }
            auto s = strides();
            size_t start = compute_flat_index(s, {index, 0});
            return tensor_ref<T, dynamic_shape>({1, shape(1)}, &data()[start], {s[0], s[1]});
            // return at({1, shape(1)}, {index, 0});
        }
        std::vector<size_t> indices(shape().size(), 0);
        assert(shape(N - 1) != 1); // Cannot end in 1
        size_t first_not_one = 0;
        // bounds check
        for (const auto &s : shape()) {
            if (s != 1) {
                assert(index < s); // Index out of bounds.
                break;
            }
            first_not_one++;
        }

        indices[first_not_one] = index;
        std::vector<size_t> new_shape(shape());
        new_shape[first_not_one] = 1;
        return at(new_shape, indices).remove_trailing();
    }
    auto operator[](size_t index) const {
        size_t N = shape().size();
        if (size() == 1) {
            // auto s = strides();
            // size_t start = compute_flat_index(s, {index});
            return tensor_ref<const T, dynamic_shape>({1}, data(), {1});
            // return at({1}, 0);
        }
        if (N == 1) {
            auto s = strides();
            size_t start = compute_flat_index(s, {index});
            return tensor_ref<const T, dynamic_shape>({1}, &data()[start], {s[0]});
            // return at({1}, {index});
        }
        if (N == 2) {
            if (shape(0) == 1) {
                auto s = strides();
                size_t start = compute_flat_index(s, {0, index});
                return tensor_ref<const T, dynamic_shape>({1}, &data()[start], {s[0]});
                // return at({1}, {0, index});
            }
            auto s = strides();
            size_t start = compute_flat_index(s, {index, 0});
            return tensor_ref<const T, dynamic_shape>({1, shape(1)}, &data()[start], {s[0], s[1]});
            // return at({1, shape(1)}, {index, 0});
        }
        std::vector<size_t> indices(shape().size(), 0);
        assert(shape(N - 1) != 1); // Cannot end in 1
        size_t first_not_one = 0;
        // bounds check
        for (const auto &s : shape()) {
            if (s != 1) {
                assert(index < s); // Index out of bounds.
                break;
            }
            first_not_one++;
        }
        indices[first_not_one] = index;
        std::vector<size_t> new_shape(shape());
        new_shape[first_not_one] = 1;
        return at(new_shape, indices).remove_trailing();
    }
    auto transpose(const std::vector<size_t> &index_permutation) {
        std::vector<size_t> sorted_perm = index_permutation;
        std::sort(sorted_perm.begin(), sorted_perm.end());
        bool duplicates = std::adjacent_find(sorted_perm.begin(), sorted_perm.end()) != sorted_perm.end();
        bool min_zero = *std::min_element(index_permutation.begin(), index_permutation.end()) == 0;
        bool size_larger = index_permutation.size() >= shape().size();
        assert(!duplicates && min_zero && size_larger); // permutation not valid
        auto old_strides = strides();
        std::vector<size_t> new_strides;
        new_strides.resize(index_permutation.size(), 1);
        std::vector<size_t> new_shape;
        new_shape.reserve(index_permutation.size());
        int i = 0;
        for (const auto &index : index_permutation) {
            if (index < old_strides.size()) {
                new_strides[i] = old_strides[index];
            }
            new_shape.push_back(shape(index));
            i++;
        }
        return tensor_ref<T, dynamic_shape>(new_shape, data(), new_strides).remove_trailing();
    }
    // specialize empty parameter list to matrix transpose
    auto transpose() { return transpose({1, 0}); }
    auto transpose(const std::vector<size_t> &index_permutation) const {
        std::vector<size_t> sorted_perm = index_permutation;
        std::sort(sorted_perm.begin(), sorted_perm.end());
        bool duplicates = std::adjacent_find(sorted_perm.begin(), sorted_perm.end()) != sorted_perm.end();
        bool min_zero = *std::min_element(index_permutation.begin(), index_permutation.end()) == 0;
        bool size_larger = index_permutation.size() >= shape().size();
        assert(!duplicates && min_zero && size_larger); // permutation not valid
        auto old_strides = strides();
        std::vector<size_t> new_strides;
        new_strides.resize(index_permutation.size(), 1);
        std::vector<size_t> new_shape;
        new_shape.reserve(index_permutation.size());
        int i = 0;
        for (const auto &index : index_permutation) {
            if (index < old_strides.size()) {
                new_strides[i] = old_strides[index];
            }
            new_shape.push_back(shape(index));
            i++;
        }
        return tensor_ref<const T, dynamic_shape>(new_shape, data(), new_strides).remove_trailing();
    }
    // specialize empty parameter list to reverse all strides
    auto transpose() const { return transpose({1, 0}); }
    auto as_ref() { return tensor_ref<T, dynamic_shape>(shape(), data(), strides()); }
    auto as_ref() const { return tensor_ref<const T, dynamic_shape>(shape(), data(), strides()); }
    void apply(T (*F)(T)) {
        for (auto &elem : *this) {
            elem = F(elem);
        }
    }
    auto remove_trailing() {
        std::vector<size_t> new_shape{};
        std::vector<size_t> new_strides{};
        auto old_strides = strides();
        size_t last_not_one = 0;
        int i = 0;
        for (const auto &size : shape()) {
            if (size != 1) {
                last_not_one = i;
            }
            i++;
        }
        if (last_not_one == shape().size() - 1) {
            return tensor_ref<T, dynamic_shape>(shape(), data(), strides());
        }
        for (int j = 0; j <= last_not_one; j++) {
            new_shape.push_back(shape(j));
            new_strides.push_back(old_strides[j]);
        }
        return tensor_ref<T, dynamic_shape>(new_shape, data(), new_strides);
    }
    auto simplify_shape() {
        std::vector<size_t> new_shape{};
        std::vector<size_t> new_strides{};
        auto old_strides = strides();
        size_t first_not_one = 0;
        size_t last_not_one = 0;
        bool first_found = false;
        int i = 0;
        for (const auto &size : shape()) {
            if (size != 1) {
                if (!first_found) {
                    first_not_one = i;
                    first_found = true;
                }
                last_not_one = i;
            }
            i++;
        }
        if (first_not_one == shape().size()) {
            return tensor_ref<T, dynamic_shape>({1}, data(), {1});
        }
        for (int j = first_not_one; j <= last_not_one; j++) {
            new_shape.push_back(shape(j));
            new_strides.push_back(old_strides[j]);
        }
        return tensor_ref<T, dynamic_shape>(new_shape, data(), new_strides);
    }
    auto remove_trailing() const {
        std::vector<size_t> new_shape{};
        std::vector<size_t> new_strides{};
        auto old_strides = strides();
        size_t last_not_one = 0;
        int i = 0;
        for (const auto &size : shape()) {
            if (size != 1) {
                last_not_one = i;
            }
            i++;
        }
        if (last_not_one == shape().size() - 1) {
            return tensor_ref<const T, dynamic_shape>(shape(), data(), strides());
        }
        for (int j = 0; j <= last_not_one; j++) {
            new_shape.push_back(shape(j));
            new_strides.push_back(old_strides[j]);
        }
        return tensor_ref<const T, dynamic_shape>(new_shape, data(), new_strides);
    }
    auto simplify_shape() const {
        std::vector<size_t> new_shape{};
        std::vector<size_t> new_strides{};
        auto old_strides = strides();
        size_t first_not_one = 0;
        size_t last_not_one = 0;
        bool first_found = false;
        int i = 0;
        for (const auto &size : shape()) {
            if (size != 1) {
                if (!first_found) {
                    first_not_one = i;
                    first_found = true;
                }
                last_not_one = i;
            }
            i++;
        }
        if (first_not_one == shape().size()) {
            return tensor_ref<const T, dynamic_shape>({1}, data(), {1});
        }
        for (int j = first_not_one; j <= last_not_one; j++) {
            new_shape.push_back(shape(j));
            new_strides.push_back(old_strides[j]);
        }
        return tensor_ref<const T, dynamic_shape>(new_shape, data(), new_strides);
    }
    auto demote_tensor(size_t index) {
        auto shape = this->shape();
        assert(shape.size() > 0);
        shape.pop_back();
        std::vector<size_t> offsets(this->shape().size(), 0);
        offsets[offsets.size() - 1] = index;
        return at(shape, offsets);
    }
    auto demote_tensor(size_t index) const {
        auto shape = this->shape();
        assert(shape.size() > 0);
        shape.pop_back();
        std::vector<size_t> offsets(this->shape().size(), 0);
        offsets[offsets.size() - 1] = index;
        return at(shape, offsets);
    }
    // produce a deep copy of the tensor
    tensor<typename std::remove_const<T>::type, dynamic_shape> copy() const {
        tensor<typename std::remove_const<T>::type, dynamic_shape> the_copy(shape(),
                                                                            typename std::remove_const<T>::type());
        std::transform(begin(), end(), the_copy.begin(),
                       [](const T &a) -> typename std::remove_const<T>::type { return a; });
        return the_copy;
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor<Q, dynamic_shape> copy_as() const {
        tensor<Q, dynamic_shape> the_copy(shape(), Q());
        std::transform(begin(), end(), the_copy.begin(), [](const T &a) -> Q { return static_cast<const Q>(a); });
        return the_copy;
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor_ref<Q, dynamic_shape> view_as() {
        return tensor_ref<Q, dynamic_shape>(shape(), reinterpret_cast<Q *>(data()), strides());
    }
    template <typename Q>
        requires(scalar<Q> || quantitative<Q>)
    tensor_ref<Q, dynamic_shape> view_as() const {
        return tensor_ref<Q, dynamic_shape>(shape(), reinterpret_cast<const Q *>(data()), strides());
    }

    // non const iterator begin. will iterate in column major order
    tensor_iterator begin() { return tensor_iterator(data(), strides(), this); }
    // non-const iterator end. will iterate in column major order
    tensor_iterator end() {
        if (shape().size() == 0) {
            return tensor_iterator(data(), strides(), this);
        }
        std::vector<size_t> indices(shape());
        for (auto &elem : indices) {
            elem--;
        }
        T *last_ptr = &data()[compute_flat_index(strides(), indices)];
        return tensor_iterator(++last_ptr, strides(), this);
    }
    // const iterator begin. will iterate in column major order
    const_tensor_iterator begin() const { return const_tensor_iterator(data(), strides(), this); }
    const_tensor_iterator cbegin() const { return begin(); }
    // const iterator end. will iterate in column major order
    const_tensor_iterator end() const {
        if (shape().size() == 0) {
            return const_tensor_iterator(data(), strides(), this);
        }
        std::vector<size_t> indices(shape());
        for (auto &elem : indices) {
            elem--;
        }
        const T *last_ptr = &data()[compute_flat_index(strides(), indices)];
        return const_tensor_iterator(++last_ptr, strides(), this);
    }
    const_tensor_iterator cend() const { return end(); }

    tensor<T, dynamic_shape> operator-() const {
        auto result = copy();
        for (auto &elem : result) {
            elem = -elem;
        }
        return result;
    }
    tensor_base &operator*=(const T &s) {
        // in case s is a scalar ref of this tensor, we need to copy to a new variable
        value_type s_copy = s;
        for (auto &elem : *this) {
            elem *= s_copy;
        }
        return *this;
    }
    tensor_base &operator/=(const T &s) {
        // in case s is a scalar ref of this tensor, we need to copy to a new variable
        value_type s_copy = s;
        for (auto &elem : *this) {
            elem /= s_copy;
        }
        return *this;
    }
    // addition and subtraction
    tensor_base &operator+=(dynamic_tensor auto const &rhs) {
        assert(rhs.shape() == shape());
        auto it2 = rhs.begin();
        for (auto it1 = begin(); it2 != rhs.end(); it1++, it2++) {
            *it1 += *it2;
        }
        return *this;
    }
    tensor_base &operator-=(dynamic_tensor auto const &rhs) {
        assert(rhs.shape() == shape());
        auto it2 = rhs.begin();
        for (auto it1 = begin(); it2 != rhs.end(); it1++, it2++) {
            *it1 -= *it2;
        }
        return *this;
    }
    tensor<T, dynamic_shape> operator+(const tensor_base &other) const {
        tensor<T, dynamic_shape> result(*this);
        result += other;
        return result;
    }
    tensor<T, dynamic_shape> operator-(const tensor_base &other) const {
        tensor<T, dynamic_shape> result(*this);
        result -= other;
        return result;
    }
    // insert flat
    template <dynamic_tensor M> void insert_flat(const M &other) {
        using V = typename M::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        assert(size() == other.size()); // tensors must have the same number of elements.
        std::transform(other.begin(), other.end(), this->begin(), [](V x) { return static_cast<T>(x); });
    }
    static tensor<T, dynamic_shape> I(size_t n) {
        auto eye = tensor<T, dynamic_shape>({n, n}, 0);
        for (size_t i = 0; i < eye.shape(0); i++) {
            eye[i][i] = 1;
        }
        return eye;
    }
    static tensor<T, dynamic_shape> zeros(const std::vector<size_t> &shape) {
        return tensor<T, dynamic_shape>(shape, 0);
    }
    static tensor<T, dynamic_shape> ones(const std::vector<size_t> &shape) {
        return tensor<T, dynamic_shape>(shape, 1);
    }
    static tensor<T, dynamic_shape> fill(const std::vector<size_t> &shape, const T &value) {
        return tensor<T, dynamic_shape>(shape, value);
    }
    static tensor<T, dynamic_shape> diag(size_t n, const T &value) {
        auto eye = tensor<T, dynamic_shape>({n, n}, 0);
        for (size_t i = 0; i < eye.shape(0); i++) {
            eye[i][i] = value;
        }
        return eye;
    }
    static tensor<T, dynamic_shape> diag(const std::vector<T> &values) {
        auto eye = tensor<T, dynamic_shape>({values.size(), values.size()}, 0);
        for (size_t i = 0; i < eye.shape(0); i++) {
            eye[i][i] = values[i];
        }
        return eye;
    }
    static tensor<T, dynamic_shape> random(const std::vector<size_t> &shape, T min = 0, T max = 1) {
        static_assert(quantitative<T> || std::is_arithmetic<T>::value, "must be arithmetic or quantitative type");
        tensor<T, dynamic_shape> result(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        for (auto &elem : result) {
            elem = dis(gen);
        }
        return result;
    }

  protected:
    std::vector<size_t> _shape{};
};
template <typename T> class tensor<T, dynamic_shape> : public tensor_base<tensor<T, dynamic_shape>, T, dynamic_shape> {

  public:
    using base_type = tensor_base<tensor<T, dynamic_shape>, T, dynamic_shape>;
    using value_type = T; /**< data type of the elements */
    tensor() : base_type({}) {}
    template <dynamic_tensor U> tensor(const U &other) : base_type(other.shape()) {
        this->_data.resize(other.size());
        std::copy(other.begin(), other.end(), this->_data.begin());
    }
    template <fixed_tensor U, int... sizes>
        requires(tensor_shape<U, sizes...> && !scalar<U>)
    tensor(const U &other) : base_type(std::vector<size_t>{sizes...}) {
        using V = typename U::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        assert(this->size() == other.size()); // tensors must have the same number of elements.
        _data.resize(this->size(), 0);
        std::transform(other.begin(), other.end(), this->begin(), [](V x) { return static_cast<T>(x); });
    }
    // Construct a new tensor filled with a constant value of the element
    tensor(const std::vector<size_t> &shape, const T &value) : base_type(shape) { _data.resize(this->size(), value); }
    // Construct a new tensor uninitialized
    tensor(const std::vector<size_t> &shape) : base_type(shape) { _data.resize(this->size()); }
    // Construct a new tensor filled with repeating blocks of the given tensor
    template <dynamic_tensor U> tensor(const std::vector<size_t> &shape, const U &other) : base_type(shape) {
        static_assert(std::is_convertible<typename U::value_type, T>::value, "Types must be convertible.");
        assert(this->check_blocks_fit(other.shape())); // Incompatible tensor sizes.
        _data.resize(this->size(), 0);
        const int count = this->size() / other.size();
        std::vector<size_t> offsets(shape.size(), 0);
        for (int i = 0; i < count; i++) {
            this->at(other.shape(), offsets) = other;
            for (int i = 0; i < shape.size(); i++) {
                if (i < other.shape().size()) {
                    offsets[i] += other.shape(i);
                } else {
                    offsets[i]++;
                }
                if (offsets[i] == this->shape(i)) {
                    offsets[i] = 0;
                } else {
                    break;
                }
            }
        }
    }
    // Construct a new tensor from a blocks of tensors or tensor_refs, blocks are stored in column major order
    template <dynamic_tensor U> tensor(std::initializer_list<U> blocks) : base_type({}) {
        using V = typename U::value_type;
        static_assert(std::is_convertible<V, T>::value, "Types must be convertible.");
        std::vector<size_t> blk_shape{};
        bool started = false;
        for (const auto &blk : blocks) {
            if (started) {
                assert(blk_shape == blk.shape()); // blocks must have matching shapes
            }
            started = true;
            blk_shape = blk.shape();
        }
        size_t n_blocks = blocks.size();
        this->_shape = blk_shape;
        this->_shape.push_back(n_blocks);
        _data.resize(this->size(), 0);
        size_t i = 0;
        for (const auto &blk : blocks) {
            for (const auto &elem : blk) {
                _data[i] = elem;
                i++;
            }
        }
    }
    // Construct a new tensor from list of values, blocks are stored in column major order
    tensor(std::initializer_list<T> values) : base_type({}) {
        this->_shape.resize(1);
        this->_shape[0] = values.size();
        _data.resize(values.size(), T(0));
        std::copy(values.begin(), values.end(), this->_data.begin());
    }
    tensor &operator=(const tensor &other) {
        _data.resize(other.size());
        this->_shape = other.shape();
        return tensor_base<tensor, T, dynamic_shape>::operator=(other);
    }
    template <dynamic_tensor M> tensor<T, dynamic_shape> &operator=(const M &other) {
        _data.resize(other.size());
        this->_shape = other.shape();
        return tensor_base<tensor, T, dynamic_shape>::operator=(other);
    }
    // assignment from scalar
    tensor<T, dynamic_shape> &operator=(const T &s) {
        assert(this->size() == 1);
        data()[0] = s;
        return *this;
    }
    // implicit conversion to T
    operator T() const {
        assert(this->size() == 1);
        return _data[0];
    }

    T *data() { return _data.data(); }
    const T *data() const { return _data.data(); }
    std::vector<size_t> strides() const { return this->compute_strides(); }
    // main tensors can be reshaped, not tensor_refs
    tensor_ref<T, dynamic_shape> as_shape(std::vector<size_t> new_shape) {
        size_t new_size = 1;
        for (const auto &size : new_shape) {
            new_size *= size;
        }
        assert(this->size() == new_size); // tensors must have the same number of elements.
        std::vector<size_t> new_strides{};
        new_strides.resize(new_shape.size());
        size_t stride = 1;
        for (size_t i = 0; i < this->shape().size(); i++) {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }
        return tensor_ref<T, dynamic_shape>(new_shape, data(), new_strides);
    }
    tensor_ref<const T, dynamic_shape> as_shape(std::vector<size_t> new_shape) const {
        size_t new_size = 1;
        for (const auto &size : new_shape) {
            new_size *= size;
        }
        assert(this->size() == new_size); // tensors must have the same number of elements.
        std::vector<size_t> new_strides{};
        new_strides.resize(new_shape.size());
        size_t stride = 1;
        for (size_t i = 0; i < this->shape().size(); i++) {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }
        return tensor_ref<const T, dynamic_shape>(new_shape, data(), new_strides);
    }
    void reshape(std::vector<size_t> new_shape) {
        size_t new_size = 1;
        for (const auto &size : new_shape) {
            new_size *= size;
        }
        if (this->size() != new_size) {
            _data.resize(new_size);
        }
        this->_shape = new_shape;
    }
    // flatten shape to a column vector
    tensor_ref<T, dynamic_shape> flatten() { return tensor_ref<T, dynamic_shape>({this->size()}, data(), {1}); }
    tensor_ref<const T, dynamic_shape> flatten() const {
        return tensor_ref<const T, dynamic_shape>({this->size()}, data(), {1});
    }

  private:
    std::vector<T> _data;
};
template <typename T>
class tensor_ref<T, dynamic_shape> : public tensor_base<tensor_ref<T, dynamic_shape>, T, dynamic_shape> {
  public:
    using base_type = tensor_base<tensor_ref<T, dynamic_shape>, T, dynamic_shape>;
    using value_type = T; /**< data type of the elements */
    tensor_ref() : _data(nullptr), _strides({}), base_type({}) {}

    // Construct a new tensor ref with a pointer to the tensor's data.
    tensor_ref(const std::vector<size_t> &shape, T *data, const std::vector<size_t> &strides)
        : _data(data), _strides(strides), base_type(shape) {}

    tensor_ref &operator=(const tensor_ref &other) {
        assert(other.shape() == this->shape());
        return tensor_base<tensor_ref, T, dynamic_shape>::operator=(other);
    }
    template <dynamic_tensor M> tensor_ref<T, dynamic_shape> &operator=(const M &other) {
        assert(other.shape() == this->shape());
        return tensor_base<tensor_ref, T, dynamic_shape>::operator=(other);
    }
    // assignment from scalar
    tensor_ref<T, dynamic_shape> &operator=(const T &s) {
        assert(this->size() == 1);
        data()[0] = s;
        return *this;
    }

    std::vector<size_t> strides() const { return _strides; }
    // implicit conversion to T&
    operator T &() {
        assert(this->size() == 1);
        return _data[0];
    }
    operator const T &() const {
        assert(this->size() == 1);
        return _data[0];
    }

    T *data() { return _data; }
    const T *data() const { return _data; }

    void set_ref(tensor_ref &ref) {
        _data = ref.data();
        _strides = ref.strides();
        base_type::_shape = ref.shape();
    }

  private:
    T *_data;                     /**< pointer to start of data being referenced*/
    std::vector<size_t> _strides; /**< strides for each dimension in the array */
};

// overload insertion operators for printing tensors
template <tensorial T>
    requires(!scalar<T>)
std::ostream &operator<<(std::ostream &os, const T &tens) {
    // auto simple_tens = tens.simplify_shape();
    size_t order = tens.order();
    if (order == 0 && tens.size() > 0) {
        os << tens.data()[0];
    } else if (order == 1) {
        os << "{";
        int i = 0;
        for (const auto &elem : tens) {
            os << elem;
            if (i != tens.size() - 1) {
                os << ", ";
            }
            i++;
        }
        os << "}";
    } else if (order == 2) {
        os << "{";
        int n = tens.shape().size();
        for (int i = 0; i < tens.shape(n - 1); i++) {
            os << tens.demote_tensor(i);
            if (i != tens.shape(n - 1) - 1) {
                os << ",\n";
            }
        }
        os << "}";
    } else if (order > 2) {
        os << "{";
        int n = tens.shape().size();
        for (int i = 0; i < tens.shape(n - 1); i++) {
            os << tens.demote_tensor(i);
            if (i != tens.shape(n - 1) - 1) {
                os << ",\n";
            }
        }
        os << "}";
    }
    return os;
}
template <scalar T> std::ostream &operator<<(std::ostream &os, const T &tens) {
    os << tens.data()[0];
    return os;
}
//  overload comparison operators for tensors
template <tensorial U, tensorial V> bool operator==(U const &x, V const &y) {
    assert(x.shape() == y.shape());
    auto it2 = y.begin();
    for (auto it1 = x.begin(); it2 != y.end(); it1++, it2++) {
        if (*it1 != *it2) {
            return false;
        }
    }
    return true;
}
template <tensorial U, tensorial V> bool operator!=(U const &x, V const &y) { return !(x == y); }

// comparison specialization for dynamic tensor scalars
template <dynamic_tensor U, dynamic_tensor V> bool operator>=(U const &x, V const &y) {
    assert(x.size() == 1 && y.size() == 1);
    return x.data()[0] >= y.data()[0];
}
template <dynamic_tensor U, dynamic_tensor V> bool operator>(U const &x, V const &y) {
    assert(x.size() == 1 && y.size() == 1);
    return x.data()[0] > y.data()[0];
}
template <dynamic_tensor U, dynamic_tensor V> bool operator<=(U const &x, V const &y) {
    assert(x.size() == 1 && y.size() == 1);
    return x.data()[0] <= y.data()[0];
}
template <dynamic_tensor U, dynamic_tensor V> bool operator<(U const &x, V const &y) {
    assert(x.size() == 1 && y.size() == 1);
    return x.data()[0] < y.data()[0];
}
using dtens = tensor<double, dynamic_shape>;
using ftens = tensor<float, dynamic_shape>;
} // namespace squint
