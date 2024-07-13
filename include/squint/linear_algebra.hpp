#ifndef SQUINT_LINEAR_ALGEBRA_HPP
#define SQUINT_LINEAR_ALGEBRA_HPP

#include "squint/dimension.hpp"
#include <numeric>
#include <utility>
#ifdef BLAS_BACKEND_MKL
#include <mkl.h>
#define BLAS_INT MKL_INT
#else
#include <cblas.h>
#include <lapacke.h>
#define BLAS_INT int
#endif

#include "squint/quantity.hpp"
#include "squint/tensor_base.hpp"
#include <algorithm>
#include <concepts>
#include <string>
#include <type_traits>

namespace squint {

// concept for scalar is arithmetic or quantitative
template <typename T>
concept scalar = arithmetic<T> || quantitative<T>;

// concept for dimensionless tensor
template <typename T>
concept dimensionless_tensor =
    tensor<T> && (arithmetic<typename T::value_type> ||
                  (quantitative<typename T::value_type> &&
                   std::is_same_v<typename T::value_type::dimension, dimensions::dimensionless>));

// checks if a compile-time list of indices has no duplicates
template <size_t... indices> constexpr bool is_unique() {
    std::array<size_t, sizeof...(indices)> arr{indices...};
    return static_cast<bool>(std::unique(arr.begin(), arr.end()) == arr.end());
}

// Concept for valid index permutation used to transpose a tensor.
// All indices must have no duplicates, and be less than the total number of indices.
template <size_t... index_permutation>
concept valid_index_permutation =
    is_unique<index_permutation...>() && ((index_permutation < sizeof...(index_permutation)) && ...);

template <std::size_t... Strides> struct strides_struct {
    static constexpr auto value = []() {
        constexpr std::size_t num_strides = sizeof...(Strides);
        std::array<std::size_t, num_strides> strides{};
        for (std::size_t i = 0; i < num_strides; ++i) {
            strides[i] = std::array<std::size_t, num_strides>{Strides...}[i];
        }
        return strides;
    }();
};

// Compile-time shape checks
template <fixed_shape_tensor A, fixed_shape_tensor B> constexpr bool compatible_for_element_wise_op() {
    auto min_dims = std::min(A::constexpr_shape().size(), B::constexpr_shape().size());
    for (std::size_t i = 0; i < min_dims; ++i) {
        if (A::constexpr_shape()[i] != B::constexpr_shape()[i]) {
            return false;
        }
    }
    return A::size() == B::size();
}

template <fixed_shape_tensor A, fixed_shape_tensor B> constexpr bool compatible_for_matmul() {
    constexpr auto this_shape = A::constexpr_shape();
    constexpr auto other_shape = B::constexpr_shape();
    constexpr bool matmat = (A::rank() == 2) && (B::rank() == 2) && (this_shape[1] == other_shape[0]);
    constexpr bool matvec = (A::rank() == 2) && (B::rank() == 1) && (this_shape[1] == other_shape[0]);
    return matmat || matvec;
}

template <tensor A, tensor B>
static constexpr bool compatible_for_blas_op()
    requires((quantitative<typename B::value_type> || arithmetic<typename B::value_type>) &&
             (quantitative<typename A::value_type> || arithmetic<typename A::value_type>))
{
    // Type compatibility check
    if constexpr (quantitative<typename A::value_type> && quantitative<typename B::value_type>) {
        static_assert(
            std::is_same_v<const typename A::value_type::value_type, const typename B::value_type::value_type>,
            "A and B underlying types must match");
    } else if constexpr (quantitative<typename A::value_type> && arithmetic<typename B::value_type>) {
        static_assert(std::is_same_v<const typename A::value_type::value_type, const typename B::value_type>,
                      "A and B underlying types must match");
    } else if constexpr (arithmetic<typename A::value_type> && quantitative<typename B::value_type>) {
        static_assert(std::is_same_v<const typename A::value_type, const typename B::value_type::value_type>,
                      "A and B underlying types must match");
    } else {
        static_assert(std::is_same_v<const typename A::value_type, const typename B::value_type>,
                      "A and B underlying types must match");
    }

    // Type checks for B
    if constexpr (quantitative<typename B::value_type>) {
        static_assert(std::is_same_v<const typename B::value_type::value_type, const float> ||
                          std::is_same_v<const typename B::value_type::value_type, const double>,
                      "B's quantity underlying type must be float or double");
    } else if constexpr (arithmetic<typename B::value_type>) {
        static_assert(std::is_same_v<const typename B::value_type, const float> ||
                          std::is_same_v<const typename B::value_type, const double>,
                      "B's tensor underlying type must be float or double");
    }

    // Type checks for A
    if constexpr (quantitative<typename A::value_type>) {
        static_assert(std::is_same_v<const typename A::value_type::value_type, const float> ||
                          std::is_same_v<const typename A::value_type::value_type, const double>,
                      "A's quantity underlying type must be float or double");
    } else if constexpr (arithmetic<typename A::value_type>) {
        static_assert(std::is_same_v<const typename A::value_type, const float> ||
                          std::is_same_v<const typename A::value_type, const double>,
                      "A's tensor underlying type must be float or double");
    }

    return true;
}

template <fixed_shape_tensor A, fixed_shape_tensor B> static constexpr bool compatible_for_solve() {
    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();

    // Shape checks
    static_assert(A::rank() == 2, "A must be 2-dimensional (matrix)");
    static_assert(B::rank() == 1 || B::rank() == 2, "B must be 1D or 2D");
    static_assert(a_shape[0] == a_shape[1], "Matrix A must be square");
    static_assert(a_shape[0] == b_shape[0], "Matrix A and vector/matrix B dimensions must match");

    // Dimensionality checks
    // A must be dimensionless since b is modified in-place to become x, therefore x must have the same units as b
    // which is only possible if A is dimensionless
    static_assert(dimensionless_tensor<A>, "A matrix must be dimensionless");

    return true;
}

template <fixed_shape_tensor A, fixed_shape_tensor B> static constexpr bool compatible_for_solve_lls() {
    // Shape checks
    static_assert(A::rank() == 2, "Matrix A must be 2-dimensional");
    static_assert(B::rank() == 1 || B::rank() == 2, "B must be 1D or 2D");

    constexpr auto x_rows = A::constexpr_shape()[1];
    constexpr auto b_rows = A::constexpr_shape()[0];
    constexpr auto max_rows = x_rows > b_rows ? x_rows : b_rows;
    static_assert(B::constexpr_shape()[0] == max_rows, "Unexpected number of rows for Matrix B");

    // Dimensionality checks
    // A must be dimensionless since b is modified in-place to become x, therefore x must have the same units as b
    // which is only possible if A is dimensionless
    static_assert(dimensionless_tensor<A>, "A matrix must be dimensionless");

    return true;
}

// compatible for cross
template <fixed_shape_tensor A, fixed_shape_tensor B> static constexpr bool compatible_for_cross() {
    static_assert(A::size() == 3 && B::size() == 3, "Cross product requires 3D vectors");
    return true;
}

// is transposed
template <fixed_shape_tensor A> static constexpr bool is_transposed() {
    constexpr auto strides = A::constexpr_strides();
    constexpr auto layout = A::get_layout();
    if constexpr (A::rank() == 2) {
        if (layout == layout::column_major) {
            return strides[0] != 1;
        }
        return strides[1] != 1;
    }
    return false;
}

// get leading dimension
template <fixed_shape_tensor A> static constexpr int get_ld() {
    constexpr auto strides = A::constexpr_strides();
    constexpr auto layout = A::get_layout();
    constexpr bool is_transposed = squint::is_transposed<A>();
    return ((layout == layout::column_major && !is_transposed) || (layout == layout::row_major && is_transposed))
               ? (A::rank() == 1 ? A::constexpr_shape()[0] : strides[1])
               : strides[0];
}

// compatible_for_inv
template <fixed_shape_tensor A> static constexpr bool compatible_for_inv() {
    static_assert(A::rank() == 2, "Inverse operation is only defined for 2D tensors (matrices)");
    constexpr auto shape = A::constexpr_shape();
    static_assert(shape[0] == shape[1], "Matrix must be square for inversion");
    return true;
}

// Helper function to build a new fixed tensor type from an existing one with a different value type
template <fixed_shape_tensor A, typename value_type, std::size_t... Is>
auto build_fixed_tensor_type(std::index_sequence<Is...> /*unused*/) {
    return fixed_tensor<value_type, A::get_layout(), A::get_error_checking(), A::constexpr_shape()[Is]...>{};
}

// Runtime shape checks
template <dynamic_shape_tensor A, dynamic_shape_tensor B> bool compatible_for_element_wise_op(const A &a, const B &b) {
    auto min_dims = std::min(a.rank(), b.rank());
    for (std::size_t i = 0; i < min_dims; ++i) {
        if (a.shape()[i] != b.shape()[i]) {
            return false;
        }
    }
    return a.size() == b.size();
}

template <dynamic_shape_tensor A, dynamic_shape_tensor B> bool compatible_for_matmul(const A &a, const B &b) {
    const auto this_shape = a.shape();
    const auto other_shape = b.shape();
    const bool matmat = (a.rank() == 2) && (b.rank() == 2) && (this_shape[1] == other_shape[0]);
    const bool matvec = (a.rank() == 2) && (b.rank() == 1) && (this_shape[1] == other_shape[0]);
    return matmat || matvec;
}

// compatible for solve
template <dynamic_shape_tensor A, dynamic_shape_tensor B> bool compatible_for_solve(const A &a, const B &b) {
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();

    // Shape checks
    if (a.rank() != 2) {
        throw std::runtime_error("Matrix A must be 2-dimensional (matrix)");
    }
    if (b.rank() != 1 && b.rank() != 2) {
        throw std::runtime_error("B must be 1D or 2D");
    }
    if (a_shape[0] != a_shape[1]) {
        throw std::runtime_error("Matrix A must be square");
    }
    if (a_shape[0] != b_shape[0]) {
        throw std::runtime_error("Matrix A and vector/matrix B dimensions must match");
    }

    // Dimensionality checks
    // A must be dimensionless since b is modified in-place to become x, therefore x must have the same units as b
    // which is only possible if A is dimensionless
    static_assert(dimensionless_tensor<A>, "A matrix must be dimensionless");

    return true;
}

// compatible for solve_lls
template <dynamic_shape_tensor A, dynamic_shape_tensor B> bool compatible_for_solve_lls(const A &a, const B &b) {
    // Shape checks
    if (a.rank() != 2) {
        throw std::runtime_error("Matrix A must be 2-dimensional");
    }
    if (b.rank() != 1 && b.rank() != 2) {
        throw std::runtime_error("B must be 1D or 2D");
    }

    const auto x_rows = a.shape()[1];
    const auto b_rows = a.shape()[0];
    const auto max_rows = x_rows > b_rows ? x_rows : b_rows;
    if (b.shape()[0] != max_rows) {
        throw std::runtime_error("Unexpected number of rows for Matrix B");
    }

    // Dimensionality checks
    // A must be dimensionless since b is modified in-place to become x, therefore x must have the same units as b
    // which is only possible if A is dimensionless
    static_assert(dimensionless_tensor<A>, "A matrix must be dimensionless");

    return true;
}

// compatible for cross
template <dynamic_shape_tensor A, dynamic_shape_tensor B> bool compatible_for_cross(const A &a, const B &b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Cross product requires 3D vectors");
    }
    return true;
}

// is transposed
template <dynamic_shape_tensor A>
bool is_transposed(const A &a, const std::vector<std::size_t> &strides, layout layout) {
    if (a.rank() == 2) {
        if (layout == layout::column_major) {
            return strides[0] != 1;
        }
        return strides[1] != 1;
    }
    return false;
}

// get leading dimension
template <dynamic_shape_tensor A>
int get_ld(const A &a, const std::vector<std::size_t> &strides, layout layout, bool is_transposed) {
    return ((layout == layout::column_major && !is_transposed) || (layout == layout::row_major && is_transposed))
               ? (a.rank() == 1 ? a.shape()[0] : strides[1])
               : strides[0];
}

// compatible_for_inv
template <dynamic_shape_tensor A> bool compatible_for_inv(const A &a) {
    if (a.rank() != 2) {
        throw std::runtime_error("Inverse operation is only defined for 2D tensors (matrices)");
    }
    const auto shape = a.shape();
    if (shape[0] != shape[1]) {
        throw std::runtime_error("Matrix must be square for inversion");
    }
    return true;
}

// Base linear algebra mixin
template <typename Derived, error_checking ErrorChecking> class linear_algebra_mixin {
  public:
    template <tensor Other> auto &operator+=(const Other &other) {
        if constexpr (fixed_shape_tensor<Derived> && fixed_shape_tensor<Other>) {
            compatible_for_element_wise_op<Derived, Other>();
        } else if constexpr (Derived::get_error_checking() == error_checking::enabled ||
                             Other::get_error_checking() == error_checking::enabled) {
            if (!compatible_for_element_wise_op(*static_cast<Derived *>(this), other)) {
                throw std::runtime_error("Incompatible shapes for element-wise operation");
            }
        }
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            *it++ += elem;
        }
        return *static_cast<Derived *>(this);
    }

    template <fixed_shape_tensor Other> auto &operator-=(const Other &other) {
        if constexpr (fixed_shape_tensor<Derived> && fixed_shape_tensor<Other>) {
            compatible_for_element_wise_op<Derived, Other>();
        } else if constexpr (Derived::get_error_checking() == error_checking::enabled ||
                             Other::get_error_checking() == error_checking::enabled) {
            if (!compatible_for_element_wise_op(*static_cast<Derived *>(this), other)) {
                throw std::runtime_error("Incompatible shapes for element-wise operation");
            }
        }
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            *it++ -= elem;
        }
        return *static_cast<Derived *>(this);
    }
    template <scalar Scalar> auto &operator*=(const Scalar &s) {
        for (auto &elem : *static_cast<Derived *>(this)) {
            elem *= s;
        }
        return *static_cast<Derived *>(this);
    }

    template <scalar Scalar> auto &operator/=(const Scalar &s) {
        for (auto &elem : *static_cast<Derived *>(this)) {
            elem /= s;
        }
        return *static_cast<Derived *>(this);
    }

    auto norm() const {
        const auto *derived = static_cast<const Derived *>(this);
        auto it = derived->begin();
        auto result = (*it++) * (*it++);
        for (std::size_t i = 1; i < derived->size(); ++i) {
            result += (*it++) * (*it++);
        }
        return squint::math::sqrt(result);
    }

    auto squared_norm() const {
        const auto *derived = static_cast<const Derived *>(this);
        auto it = derived->begin();
        auto result = (*it++) * (*it++);
        for (std::size_t i = 1; i < derived->size(); ++i) {
            result += (*it++) * (*it++);
        }
        return result;
    }

    auto trace() const {
        const auto *derived = static_cast<const Derived *>(this);
        if constexpr (fixed_shape_tensor<Derived>) {
            if constexpr (Derived::rank() == 2) {
                if (derived->shape()[0] != derived->shape()[1]) {
                    throw std::runtime_error("Matrix must be square for trace");
                }
            }
        }
        if constexpr (dynamic_shape_tensor<Derived>) {
            if constexpr (Derived::get_error_checking() == error_checking::enabled) {
                if (derived->shape()[0] != derived->shape()[1]) {
                    throw std::runtime_error("Matrix must be square for trace");
                }
            }
        }
        auto it = derived->begin();
        auto result = *it++;
        for (std::size_t i = 1; i < derived->shape()[0]; ++i) {
            result += *it;
            it += derived->strides()[0] + 1;
        }
        return result;
    }

    auto mean() const { return sum() / static_cast<const Derived *>(this)->size(); }

    auto sum() const {
        const auto *derived = static_cast<const Derived *>(this);
        return std::accumulate(derived->begin(), derived->end(), typename Derived::value_type{});
    }

    auto min() const {
        const auto *derived = static_cast<const Derived *>(this);
        return *std::min_element(derived->begin(), derived->end());
    }

    auto max() const {
        const auto *derived = static_cast<const Derived *>(this);
        return *std::max_element(derived->begin(), derived->end());
    }
};

// Fixed tensor with linear algebra
template <typename Derived, error_checking ErrorChecking>
class fixed_linear_algebra_mixin : public linear_algebra_mixin<Derived, ErrorChecking> {
  public:
    template <fixed_shape_tensor B> auto operator/(const B &b) const {
        auto derived = static_cast<const Derived *>(this);
        // Solve the general linear least squares problem
        // make a copy of A since it will be modified
        // TODO
        static_assert(false);
        using value_type = decltype(derived->raw_data()[0]);
        auto A =
            build_fixed_tensor_type<Derived, value_type>(std::make_index_sequence<Derived::constexpr_shape().size()>{});
        auto it = derived->begin();
        for (auto &elem : A) {
            elem = value_type(*it++);
        }
        using x_type = decltype(typename B::value_type{} / typename Derived::value_type{});
        if constexpr (B::constexpr_shape().size() == 1) {
            fixed_tensor<x_type, Derived::get_layout(), Derived::get_error_checking(), Derived::constexpr_shape()[1]> x;
            solve_lls(A, b, x);
            return x;
        } else {
            fixed_tensor<x_type, Derived::get_layout(), Derived::get_error_checking(), Derived::constexpr_shape()[1],
                         B::constexpr_shape()[1]>
                x;
            solve_lls(A, b, x);
            return x;
        }
    }

    template <fixed_shape_tensor Other> bool operator==(const Other &other) const {
        compatible_for_element_wise_op<Derived, Other>();
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            if (*it++ != elem) {
                return false;
            }
        }
        return true;
    }

    template <fixed_shape_tensor Other> bool operator!=(const Other &other) const { return !(*this == other); }

    template <size_t... index_permutation>
    auto transpose()
        requires valid_index_permutation<index_permutation...>
    {
        static_assert(sizeof...(index_permutation) == Derived::rank(),
                      "Number of indices must match the rank of the tensor");
        auto derived = static_cast<Derived *>(this);
        constexpr size_t N = sizeof...(index_permutation);
        return [&]<auto... I>(std::index_sequence<I...>) {
            using strides = strides_struct<Derived::constexpr_strides()[index_permutation]...>;
            return fixed_tensor_view<typename Derived::value_type, Derived::get_layout(), strides,
                                     Derived::get_error_checking(), Derived::constexpr_shape()[index_permutation]...>(
                derived->data());
        }(std::make_index_sequence<N>{});
    }

    template <size_t... index_permutation>
    auto transpose() const
        requires valid_index_permutation<index_permutation...>
    {
        static_assert(sizeof...(index_permutation) == Derived::rank(),
                      "Number of indices must match the rank of the tensor");
        const auto derived = static_cast<const Derived *>(this);
        constexpr size_t N = sizeof...(index_permutation);
        return [&]<auto... I>(std::index_sequence<I...>) {
            using strides = strides_struct<Derived::constexpr_strides()[index_permutation]...>;
            return const_fixed_tensor_view<typename Derived::value_type, Derived::get_layout(), strides,
                                           Derived::get_error_checking(),
                                           Derived::constexpr_shape()[index_permutation]...>(derived->data());
        }(std::make_index_sequence<N>{});
    }

    auto transpose() {
        static_assert(Derived::rank() > 0, "Cannot transpose a scalar");
        static_assert(Derived::rank() < 3, "Specify the permutation of indices for tensors with rank > 2");
        // Specialization for 2D tensors
        if constexpr (Derived::rank() == 2) {
            return static_cast<Derived *>(this)->template transpose<1, 0>();
        }
        // Specialization for 1D tensors
        if constexpr (Derived::rank() == 1) {
            return static_cast<Derived *>(this)->template reshape<1, Derived::constexpr_shape()[0]>();
        }
    }
    auto transpose() const {
        static_assert(Derived::rank() > 0, "Cannot transpose a scalar");
        static_assert(Derived::rank() < 3, "Specify the permutation of indices for tensors with rank > 2");
        // Specialization for 2D tensors
        if constexpr (Derived::rank() == 2) {
            return static_cast<const Derived *>(this)->template transpose<1, 0>();
        }
        // Specialization for 1D tensors
        if constexpr (Derived::rank() == 1) {
            return static_cast<const Derived *>(this)->template reshape<1, Derived::constexpr_shape()[0]>();
        }
    }

    auto inv() const {
        compatible_for_inv<Derived>();
        compatible_for_blas_op<Derived, Derived>();

        const auto *derived = static_cast<const Derived *>(this);
        constexpr auto shape = Derived::constexpr_shape();

        using result_value_type = decltype(1 / typename Derived::value_type{});
        auto result = build_fixed_tensor_type<Derived, result_value_type>(std::make_index_sequence<2>{});

        constexpr int n = shape[0];
        constexpr int lda = get_ld<Derived>();

        std::vector<BLAS_INT> ipiv(n);

        // Determine LAPACK layout
        constexpr int lapack_layout = (Derived::constexpr_shape()[0] != 1) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

        int info;

        // Copy data to result
        auto it = derived->begin();
        for (auto &elem : result) {
            elem = result_value_type(*it++);
        }
        using blas_type = decltype(derived->raw_data()[0]);

        // LU decomposition
        if constexpr (std::is_same_v<blas_type, const float &>) {
            info = LAPACKE_sgetrf(lapack_layout, n, n, result.raw_data(), lda, ipiv.data());
        } else {
            info = LAPACKE_dgetrf(lapack_layout, n, n, result.raw_data(), lda, ipiv.data());
        }

        if (info != 0) {
            throw std::runtime_error("LU decomposition failed with error code " + std::to_string(info));
        }

        // Matrix inversion
        if constexpr (std::is_same_v<blas_type, const float &>) {
            info = LAPACKE_sgetri(lapack_layout, n, result.raw_data(), lda, ipiv.data());
        } else {
            info = LAPACKE_dgetri(lapack_layout, n, result.raw_data(), lda, ipiv.data());
        }

        if (info != 0) {
            throw std::runtime_error("Matrix inversion failed with error code " + std::to_string(info));
        }

        return result;
    }

    auto pinv() const {
        const auto *derived = static_cast<const Derived *>(this);
        constexpr auto shape = Derived::constexpr_shape();
        constexpr int m = shape[0];
        constexpr int n = shape[1];

        if constexpr (m >= n) {
            // Overdetermined or square system: pinv(A) = (A^T * A)^-1 * A^T
            auto AtA = derived->transpose() * (*derived);
            return AtA.inv() * derived->transpose();
        } else {
            // Underdetermined system: pinv(A) = A^T * (A * A^T)^-1
            auto AAt = (*derived) * derived->transpose();
            return derived->transpose() * AAt.inv();
        }
    }
};

// Element-wise operations between fixed tensors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator+(const A &a, const B &b) {
    compatible_for_element_wise_op<A, B>();
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ += elem;
    }
    return result;
}

template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator-(const A &a, const B &b) {
    compatible_for_element_wise_op<A, B>();
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ -= elem;
    }
    return result;
}

// Matrix multiplication between fixed tensors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator*(const A &a, const B &b) {
    compatible_for_matmul<A, B>();
    compatible_for_blas_op<A, B>();
    // layouts must match
    static_assert(A::get_layout() == B::get_layout(), "A and B layouts must match");
    // matmul using BLAS
    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();
    constexpr auto layout = A::get_layout();

    constexpr int m = a_shape[0];
    constexpr int n = b_shape[1];
    constexpr int k = a_shape[1];

    // Determine if matrix is transposed
    constexpr auto op_a = is_transposed<A>() ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    constexpr auto op_b = is_transposed<B>() ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    // Determine leading dimensions based on layout and transposition
    constexpr int lda = get_ld<A>();
    constexpr int ldb = get_ld<B>();
    constexpr int ldc = m;
    auto result = fixed_tensor<decltype(typename A::value_type{} * typename B::value_type{}), layout,
                               A::get_error_checking(), m, n>();

    // Determine BLAS layout
    constexpr auto blas_layout =
        (layout == layout::row_major) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;

    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        cblas_sgemm(blas_layout, op_a, op_b, m, n, k, 1.0F, const_cast<float *>(a.raw_data()), lda,
                    const_cast<float *>(b.raw_data()), ldb, 0.0F, result.raw_data(), ldc);
    } else {
        cblas_dgemm(blas_layout, op_a, op_b, m, n, k, 1.0, const_cast<double *>(a.raw_data()), lda,
                    const_cast<double *>(b.raw_data()), ldb, 0.0, result.raw_data(), ldc);
    }

    return result;
}

// Solve linear system of equations Ax = b
template <fixed_shape_tensor A, fixed_shape_tensor B> auto solve(A &a, B &b) {
    compatible_for_solve<A, B>();
    compatible_for_blas_op<A, B>();
    static_assert(A::get_layout() == B::get_layout(), "A and B layouts must match");
    static_assert(!is_transposed<A>() && !is_transposed<B>(), "A and B must not be transposed");

    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();
    constexpr auto layout = A::get_layout();

    constexpr int n = a_shape[0];
    constexpr int nrhs = (B::rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    constexpr int lda = get_ld<A>();
    constexpr int ldb = get_ld<B>();

    std::vector<BLAS_INT> ipiv(n);

    // Determine LAPACK layout
    int lapack_layout = (layout == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

    int info;
    if constexpr (std::is_same_v<decltype(b.raw_data()), float *> ||
                  std::is_same_v<decltype(b.raw_data()), const float *>) {
        info = LAPACKE_sgesv(lapack_layout, n, nrhs, const_cast<float *>(a.raw_data()), lda, ipiv.data(), b.raw_data(),
                             ldb);
    } else {
        info = LAPACKE_dgesv(lapack_layout, n, nrhs, const_cast<double *>(a.raw_data()), lda, ipiv.data(), b.raw_data(),
                             ldb);
    }

    if (info != 0) {
        throw std::runtime_error("LAPACKE_gesv failed with error code " + std::to_string(info));
    }
    return ipiv;
}

// Solve linear least squares problem Ax = b or the minimum norm solution
template <fixed_shape_tensor A, fixed_shape_tensor B> void solve_lls(A &a, B &b) {
    compatible_for_solve_lls<A, B>();
    compatible_for_blas_op<A, B>();
    static_assert(A::get_layout() == B::get_layout(), "A and B layouts must match");
    static_assert(!is_transposed<A>() && !is_transposed<B>(), "A and B must not be transposed");

    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();
    constexpr auto layout = A::get_layout();

    constexpr int m = a_shape[0];
    constexpr int n = a_shape[1];
    constexpr int nrhs = (B::rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    constexpr int lda = get_ld<A>();
    constexpr int ldb = get_ld<B>();

    // Determine LAPACK layout
    int lapack_layout = (layout == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

    auto trans = is_transposed<A>() ? 'T' : 'N';

    int info;
    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        info =
            LAPACKE_sgels(lapack_layout, trans, m, n, nrhs, const_cast<float *>(a.raw_data()), lda, b.raw_data(), ldb);
    } else {
        info =
            LAPACKE_dgels(lapack_layout, trans, m, n, nrhs, const_cast<double *>(a.raw_data()), lda, b.raw_data(), ldb);
    }

    if (info != 0) {
        throw std::runtime_error("LAPACKE_gels failed with error code " + std::to_string(info));
    }
}

// Cross product between two 3D vectors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto cross(const A &a, const B &b) {
    compatible_for_cross<A, B>();
    using result_value_type = decltype(typename A::value_type{} * typename B::value_type{});
    auto result = fixed_tensor<result_value_type, A::get_layout(), A::get_error_checking(), 3>();

    // get flat iterator for both tensors
    auto a_it = a.begin();
    auto b_it = b.begin();

    // Calculate cross product (used iterators since shapes may not be one dimensional)
    result[0] = (*a_it)[1] * (*b_it)[2] - (*a_it)[2] * (*b_it)[1];
    result[1] = (*a_it)[2] * (*b_it)[0] - (*a_it)[0] * (*b_it)[2];
    result[2] = (*a_it)[0] * (*b_it)[1] - (*a_it)[1] * (*b_it)[0];

    return result;
}

template <fixed_shape_tensor A, fixed_shape_tensor B, typename Epsilon>
bool approx_equal(const A &a, const B &b, const Epsilon &epsilon) {
    compatible_for_element_wise_op<A, B>();
    auto it = a.begin();
    for (const auto &elem : b) {
        if (!approx_equal(*it++, elem, epsilon)) {
            return false;
        }
    }
    return true;
}

// Scalar multiplication
template <fixed_shape_tensor A, scalar Scalar> auto operator*(const A &a, const Scalar &s) {
    using result_value_type = decltype(typename A::value_type{} * s);
    auto result = build_fixed_tensor_type<A, result_value_type>(std::make_index_sequence<A::rank()>{});
    auto a_iter = a.begin();
    for (auto &elem : result) {
        elem = (*a_iter++) * s;
    }
    return result;
}
template <fixed_shape_tensor A, scalar Scalar> auto operator*(const Scalar &s, const A &a) { return a * s; }

// Scalar division
template <fixed_shape_tensor A, scalar Scalar> auto operator/(const A &a, const Scalar &s) {
    using result_value_type = decltype(typename A::value_type{} / s);
    auto result = build_fixed_tensor_type<A, result_value_type>(std::make_index_sequence<A::rank()>{});
    auto a_iter = a.begin();
    for (auto &elem : result) {
        elem = (*a_iter++) / s;
    }
    return result;
}

// Dynamic tensor with linear algebra
template <typename Derived, error_checking ErrorChecking>
class dynamic_linear_algebra_mixin : public linear_algebra_mixin<Derived, ErrorChecking> {
  public:
    template <tensor B> auto operator/(const B &b) const {
        // solve the general linear least squares problem
        // make a copy of A since it will be modified
        auto A = *static_cast<const Derived *>(this);
        using x_type = decltype(typename B::value_type{} / typename Derived::value_type{});
        if (b.rank() == 1) {
            dynamic_tensor<x_type, Derived::get_error_checking()> x({A.shape()[1]});
            solve_lls(A, b, x);
            return x;
        }
        dynamic_tensor<x_type, Derived::get_error_checking()> x({A.shape()[1], b.shape()[1]});
        solve_lls(A, b, x);
        return x;
    }
    template <tensor Other> bool operator==(const Other &other) const {
        if constexpr (Derived::get_error_checking() == error_checking::enabled ||
                      Other::get_error_checking() == error_checking::enabled) {
            if (!compatible_for_element_wise_op(*static_cast<const Derived *>(this), other)) {
                throw std::runtime_error("Incompatible shapes for element-wise operation");
            }
        }
        auto it = static_cast<const Derived *>(this)->begin();
        for (const auto &elem : other) {
            if (*it++ != elem) {
                return false;
            }
        }
        return true;
    }
    template <tensor Other> bool operator!=(const Other &other) const { return !(*this == other); }

    auto transpose(std::vector<std::size_t> index_permutation) {
        if constexpr (Derived::get_error_checking() == error_checking::enabled) {
            if (index_permutation.size() != Derived::rank()) {
                throw std::runtime_error("Number of indices must match the rank of the tensor");
            }
            if (!std::is_permutation(index_permutation.begin(), index_permutation.end(),
                                     std::make_index_sequence<Derived::rank()>::begin())) {
                throw std::runtime_error("Invalid index permutation");
            }
        }
        auto derived = static_cast<Derived *>(this);
        auto strides = derived->strides();
        for (std::size_t i = 0; i < index_permutation.size(); ++i) {
            strides[i] = derived->strides()[index_permutation[i]];
        }
        return dynamic_tensor_view<typename Derived::value_type, Derived::get_error_checking()>(
            derived->data(), derived->shape(), strides, derived->get_layout());
    }
    auto transpose(std::vector<std::size_t> index_permutation) const {
        if constexpr (Derived::get_error_checking() == error_checking::enabled) {
            if (index_permutation.size() != Derived::rank()) {
                throw std::runtime_error("Number of indices must match the rank of the tensor");
            }
            if (!std::is_permutation(index_permutation.begin(), index_permutation.end(),
                                     std::make_index_sequence<Derived::rank()>::begin())) {
                throw std::runtime_error("Invalid index permutation");
            }
        }
        const auto derived = static_cast<const Derived *>(this);
        auto strides = derived->strides();
        for (std::size_t i = 0; i < index_permutation.size(); ++i) {
            strides[i] = derived->strides()[index_permutation[i]];
        }
        return const_dynamic_tensor_view<typename Derived::value_type, Derived::get_error_checking()>(
            derived->data(), derived->shape(), strides, derived->get_layout());
    }

    auto transpose() {
        auto derived = static_cast<Derived *>(this);
        if (derived->rank() > 0) {
            if (derived->rank() == 2) {
                return derived->transpose({1, 0});
            }
            return derived->transpose({1});
        }
        throw std::runtime_error("Specifying the permutation of indices is required for tensors with rank > 2");
    }

    auto transpose() const {
        const auto derived = static_cast<const Derived *>(this);
        if (derived->rank() > 0) {
            if (derived->rank() == 2) {
                return derived->transpose({1, 0});
            }
            return derived->transpose({1});
        }
        throw std::runtime_error("Specifying the permutation of indices is required for tensors with rank > 2");
    }

    auto inv() const {
        const auto *derived = static_cast<const Derived *>(this);
        if constexpr (Derived::get_error_checking() == error_checking::enabled) {
            compatible_for_inv(*derived);
        }
        compatible_for_blas_op<Derived, Derived>();

        using result_value_type = decltype(1 / typename Derived::value_type{});
        auto result = dynamic_tensor<result_value_type, Derived::get_error_checking()>(
            {derived->shape()[0], derived->shape()[1]});

        const int n = derived->shape()[0];
        const int lda = get_ld(*derived, derived->strides(), derived->get_layout(),
                               is_transposed(*derived, derived->strides(), derived->get_layout()));

        std::vector<BLAS_INT> ipiv(n);

        int lapack_layout = (derived->get_layout() == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

        int info;

        auto it = derived->begin();
        for (auto &elem : result) {
            elem = result_value_type(*it++);
        }

        using blas_type = decltype(derived->raw_data()[0]);

        if constexpr (std::is_same_v<blas_type, const float &>) {
            info = LAPACKE_sgetrf(lapack_layout, n, n, result.raw_data(), lda, ipiv.data());
        } else {
            info = LAPACKE_dgetrf(lapack_layout, n, n, result.raw_data(), lda, ipiv.data());
        }
        if (info != 0) {
            throw std::runtime_error("LU decomposition failed with error code " + std::to_string(info));
        }
        if constexpr (std::is_same_v<blas_type, const float &>) {
            info = LAPACKE_sgetri(lapack_layout, n, result.raw_data(), lda, ipiv.data());
        } else {
            info = LAPACKE_dgetri(lapack_layout, n, result.raw_data(), lda, ipiv.data());
        }
        if (info != 0) {
            throw std::runtime_error("Matrix inversion failed with error code " + std::to_string(info));
        }
        return result;
    }

    auto pinv() const {
        const auto *derived = static_cast<const Derived *>(this);
        const auto &shape = derived->shape();
        const int m = shape[0];
        const int n = shape[1];

        if (m >= n) {
            // Overdetermined or square system: pinv(A) = (A^T * A)^-1 * A^T
            auto AtA = derived->transpose() * (*derived);
            return AtA.inv() * derived->transpose();
        }
        // Underdetermined system: pinv(A) = A^T * (A * A^T)^-1
        auto AAt = (*derived) * derived->transpose();
        return derived->transpose() * AAt.inv();
    }
};

// Element-wise operations between dynamic tensors
template <dynamic_shape_tensor A, dynamic_shape_tensor B> auto operator+(const A &a, const B &b) {
    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_element_wise_op(a, b)) {
            throw std::runtime_error("Incompatible shapes for element-wise operation");
        }
    }
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ += elem;
    }
    return result;
}

template <dynamic_shape_tensor A, dynamic_shape_tensor B> auto operator-(const A &a, const B &b) {
    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_element_wise_op(a, b)) {
            throw std::runtime_error("Incompatible shapes for element-wise operation");
        }
    }
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ -= elem;
    }
    return result;
}

// Matrix multiplication between dynamic tensors
template <dynamic_shape_tensor A, dynamic_shape_tensor B> auto operator*(const A &a, const B &b) {
    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_matmul(a, b)) {
            throw std::runtime_error("Incompatible shapes for matrix multiplication");
        }
        // layouts must match
        if (a.get_layout() != a.get_layout()) {
            throw std::runtime_error("A and B layouts must match");
        }
    }
    compatible_for_blas_op<A, B>();

    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const auto a_strides = a.strides();
    const auto b_strides = b.strides();
    const auto layout = a.get_layout();

    const int m = a_shape[0];
    const int n = b_shape[1];
    const int k = a_shape[1];

    // Determine if matrix is transposed based on strides and layout
    // if column major and first stride isn't 1, then matrix is transposed
    // if row major and last stride isn't 1, then matrix is transposed
    const auto op_a = is_transposed(a, a_strides, layout) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
    const auto op_b = is_transposed(b, b_strides, layout) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

    // Determine leading dimensions based on layout and op
    const int lda = get_ld(a, a_strides, layout, op_a == CBLAS_TRANSPOSE::CblasTrans);
    const int ldb = get_ld(b, b_strides, layout, op_b == CBLAS_TRANSPOSE::CblasTrans);
    const int ldc = m;

    auto result =
        dynamic_tensor<decltype(typename A::value_type{} * typename B::value_type{}), A::get_error_checking()>(
            {std::size_t(m), std::size_t(n)}, a.get_layout());

    // Determine BLAS layout
    auto blas_layout = (a.get_layout() == layout::row_major) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;

    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        cblas_sgemm(blas_layout, op_a, op_b, m, n, k, 1.0F, const_cast<float *>(a.raw_data()), lda,
                    const_cast<float *>(b.raw_data()), ldb, 0.0F, result.raw_data(), ldc);
    } else {
        cblas_dgemm(blas_layout, op_a, op_b, m, n, k, 1.0, const_cast<double *>(a.raw_data()), lda,
                    const_cast<double *>(b.raw_data()), ldb, 0.0, result.raw_data(), ldc);
    }

    return result;
}

// Solve linear system of equations Ax = b
template <dynamic_shape_tensor A, dynamic_shape_tensor B> auto solve(A &a, B &b) {
    const auto a_strides = a.strides();
    const auto b_strides = b.strides();
    const auto layout = a.get_layout();
    const bool a_is_transposed = is_transposed(a, a_strides, layout);
    const bool b_is_transposed = is_transposed(b, b_strides, layout);

    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_solve(a, b)) {
            throw std::runtime_error("Incompatible shapes for solving linear system");
        }
        // layouts must match
        if (a.get_layout() != b.get_layout()) {
            throw std::runtime_error("A and B layouts must match");
        }
        // must not be transposed
        if (a_is_transposed || b_is_transposed) {
            throw std::runtime_error("A and B must not be transposed");
        }
    }
    compatible_for_blas_op<A, B>();

    const auto a_shape = a.shape();
    const auto b_shape = b.shape();

    const int n = a_shape[0];
    const int nrhs = (b.rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    const int lda = get_ld(a, a_strides, layout, a_is_transposed);
    const int ldb = get_ld(b, b_strides, layout, b_is_transposed);

    std::vector<BLAS_INT> ipiv(n);

    // Determine LAPACK layout
    int lapack_layout = (layout == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

    int info;
    if constexpr (std::is_same_v<decltype(b.raw_data()), float *> ||
                  std::is_same_v<decltype(b.raw_data()), const float *>) {
        info = LAPACKE_sgesv(lapack_layout, n, nrhs, const_cast<float *>(a.raw_data()), lda, ipiv.data(),
                             const_cast<float *>(b.raw_data()), ldb);
    } else {
        info = LAPACKE_dgesv(lapack_layout, n, nrhs, const_cast<double *>(a.raw_data()), lda, ipiv.data(),
                             const_cast<double *>(b.raw_data()), ldb);
    }

    if (info != 0) {
        throw std::runtime_error("LAPACKE_gesv failed with error code " + std::to_string(info));
    }
    return ipiv;
}

// Solve linear least squares problem Ax = b
template <dynamic_shape_tensor A, dynamic_shape_tensor B> void solve_lls(A &a, B &b) {
    const auto a_strides = a.strides();
    const auto b_strides = b.strides();
    const auto layout = a.get_layout();
    const bool a_is_transposed = is_transposed(a, a_strides, layout);
    const bool b_is_transposed = is_transposed(b, b_strides, layout);

    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_solve_lls(a, b)) {
            throw std::runtime_error("Incompatible shapes for solving linear least squares");
        }
        // layouts must match
        if (a.get_layout() != b.get_layout()) {
            throw std::runtime_error("A and B layouts must match");
        }
        // must not be transposed
        if (a_is_transposed || b_is_transposed) {
            throw std::runtime_error("A and B must not be transposed");
        }
    }
    compatible_for_blas_op<A, B>();

    const auto a_shape = a.shape();
    const auto b_shape = b.shape();

    const int m = a_shape[0];
    const int n = a_shape[1];
    const int nrhs = (b.rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    const int lda = get_ld(a, a_strides, layout, a_is_transposed);
    const int ldb = get_ld(b, b_strides, layout, b_is_transposed);

    // Determine LAPACK layout
    int lapack_layout = (layout == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

    int info;
    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        info = LAPACKE_sgels(lapack_layout, 'N', m, n, nrhs, const_cast<float *>(a.raw_data()), lda, b.raw_data(), ldb);
    } else {
        info =
            LAPACKE_dgels(lapack_layout, 'N', m, n, nrhs, const_cast<double *>(a.raw_data()), lda, b.raw_data(), ldb);
    }

    if (info) {
        throw std::runtime_error("LAPACKE_gels failed with error code " + std::to_string(info));
    }
}

// cross product between two 3D vectors
template <dynamic_shape_tensor A, dynamic_shape_tensor B> auto cross(const A &a, const B &b) {
    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_cross(a, b)) {
            throw std::runtime_error("Incompatible shapes for cross product");
        }
    }
    using result_value_type = decltype(typename A::value_type{} * typename B::value_type{});
    auto result = dynamic_tensor<result_value_type, A::get_error_checking()>({3});

    // get flat iterator for both tensors
    auto a_it = a.begin();
    auto b_it = b.begin();

    // Calculate cross product (used iterators since shapes may not be one dimensional)
    result[0] = (*a_it)[1] * (*b_it)[2] - (*a_it)[2] * (*b_it)[1];
    result[1] = (*a_it)[2] * (*b_it)[0] - (*a_it)[0] * (*b_it)[2];
    result[2] = (*a_it)[0] * (*b_it)[1] - (*a_it)[1] * (*b_it)[0];

    return result;
}

// approx_equal for dynamic tensors
template <dynamic_shape_tensor A, dynamic_shape_tensor B, typename Epsilon>
bool approx_equal(const A &a, const B &b, const Epsilon &epsilon) {
    if constexpr (A::get_error_checking() == error_checking::enabled ||
                  B::get_error_checking() == error_checking::enabled) {
        if (!compatible_for_element_wise_op(a, b)) {
            throw std::runtime_error("Incompatible shapes for element-wise operation");
        }
    }
    auto it = a.begin();
    for (const auto &elem : b) {
        if (!approx_equal(*it++, elem, epsilon)) {
            return false;
        }
    }
    return true;
}

// Scalar multiplication
template <dynamic_shape_tensor A, scalar Scalar> auto operator*(const A &a, const Scalar &s) {
    using result_value_type = decltype(typename A::value_type{} * s);
    auto result = dynamic_tensor<result_value_type, A::get_error_checking()>(a.shape(), a.get_layout());
    auto a_iter = a.begin();
    for (auto &elem : result) {
        elem = (*a_iter++) * s;
    }
    return result;
}

template <dynamic_shape_tensor A, scalar Scalar> auto operator*(const Scalar &s, const A &a) { return a * s; }

// Scalar division
template <dynamic_shape_tensor A, scalar Scalar> auto operator/(const A &a, const Scalar &s) {
    using result_value_type = decltype(typename A::value_type{} / s);
    auto result = dynamic_tensor<result_value_type, A::get_error_checking()>(a.shape(), a.get_layout());
    auto a_iter = a.begin();
    for (auto &elem : result) {
        elem = (*a_iter++) / s;
    }
    return result;
}

} // namespace squint

#endif // SQUINT_LINEAR_ALGEBRA_HPP