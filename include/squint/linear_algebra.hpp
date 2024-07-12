#ifndef SQUINT_LINEAR_ALGEBRA_HPP
#define SQUINT_LINEAR_ALGEBRA_HPP

#include "squint/dimension.hpp"
#include <numeric>
#ifdef BLAS_BACKEND_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "squint/quantity.hpp"
#include "squint/tensor_base.hpp"
#include <algorithm>
#include <type_traits>

namespace squint {

// Base linear algebra mixin
template <typename Derived, error_checking ErrorChecking> class linear_algebra_mixin {
  public:
    auto transpose();
    auto transpose() const;
    auto determinant() const;
    auto inv() const;
    auto norm(int p = 2) const;
    auto trace() const;
    auto eigenvalues() const;
    auto eigenvectors() const;
    auto pinv() const;

    auto mean() const { return sum() / static_cast<const Derived *>(this)->size(); }

    auto sum() const {
        // TODO, can be optimized with BLAS algorithms
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

template <fixed_shape_tensor A, fixed_shape_tensor B>
static constexpr bool compatible_for_type()
    requires((quantitative<typename B::value_type> || arithmetic<typename B::value_type>) &&
             (quantitative<typename A::value_type> || arithmetic<typename A::value_type>))
{
    // Type compatibility check
    if constexpr (quantitative<typename A::value_type> && quantitative<typename B::value_type>) {
        static_assert(std::is_same_v<typename A::value_type::value_type, typename B::value_type::value_type>,
                      "A and B underlying types must match");
    } else if constexpr (quantitative<typename A::value_type> && arithmetic<typename B::value_type>) {
        static_assert(std::is_same_v<typename A::value_type::value_type, typename B::value_type>,
                      "A and B underlying types must match");
    } else if constexpr (arithmetic<typename A::value_type> && quantitative<typename B::value_type>) {
        static_assert(std::is_same_v<typename A::value_type, typename B::value_type::value_type>,
                      "A and B underlying types must match");
    } else {
        static_assert(std::is_same_v<typename A::value_type, typename B::value_type>,
                      "A and B underlying types must match");
    }

    return true;
}

template <fixed_shape_tensor A, fixed_shape_tensor B>
static constexpr bool compatible_for_solve()
    requires((quantitative<typename B::value_type> || arithmetic<typename B::value_type>) &&
             (quantitative<typename A::value_type> || arithmetic<typename A::value_type>))
{
    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();

    // Shape checks
    static_assert(A::rank() == 2, "A must be 2-dimensional (matrix)");
    static_assert(B::rank() == 1 || B::rank() == 2, "B must be 1D or 2D");
    static_assert(a_shape[0] == a_shape[1], "Matrix A must be square");
    static_assert(a_shape[0] == b_shape[0], "Matrix A and vector/matrix B dimensions must match");

    // Type checks for B
    if constexpr (quantitative<typename B::value_type>) {
        static_assert(std::is_same_v<typename B::value_type::value_type, float> ||
                          std::is_same_v<typename B::value_type::value_type, double>,
                      "B's quantity underlying type must be float or double");
    } else if constexpr (arithmetic<typename B::value_type>) {
        static_assert(std::is_same_v<typename B::value_type, float> || std::is_same_v<typename B::value_type, double>,
                      "B's tensor underlying type must be float or double");
    }

    // Type checks for A
    if constexpr (quantitative<typename A::value_type>) {
        static_assert(std::is_same_v<typename A::value_type::dimension_type, dimensions::dimensionless>,
                      "A's quantity type must be dimensionless");
        static_assert(std::is_same_v<typename A::value_type::value_type, float> ||
                          std::is_same_v<typename A::value_type::value_type, double>,
                      "A's quantity underlying type must be float or double");
    } else if constexpr (arithmetic<typename A::value_type>) {
        static_assert(std::is_same_v<typename A::value_type, float> || std::is_same_v<typename A::value_type, double>,
                      "A's tensor underlying type must be float or double");
    }

    // Type compatibility check
    compatible_for_type<A, B>();

    return true;
}

// Fixed tensor with linear algebra
template <typename Derived, error_checking ErrorChecking>
class fixed_linear_algebra_mixin : public linear_algebra_mixin<Derived, ErrorChecking> {
  public:
    template <typename Scalar> auto &operator*=(const Scalar &scalar) {
        for (auto &elem : *static_cast<Derived *>(this)) {
            elem *= scalar;
        }
        return *static_cast<Derived *>(this);
    }

    template <typename Scalar> auto &operator/=(const Scalar &scalar) {
        for (auto &elem : *static_cast<Derived *>(this)) {
            elem /= scalar;
        }
        return *static_cast<Derived *>(this);
    }

    template <fixed_shape_tensor Other> auto &operator+=(const Other &other) {
        static_assert(compatible_for_element_wise_op<Derived, Other>(),
                      "Incompatible shapes for element-wise addition");
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            *it++ += elem;
        }
        return *static_cast<Derived *>(this);
    }

    template <fixed_shape_tensor Other> auto &operator-=(const Other &other) {
        static_assert(compatible_for_element_wise_op<Derived, Other>(),
                      "Incompatible shapes for element-wise subtraction");
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            *it++ -= elem;
        }
        return *static_cast<Derived *>(this);
    }

    template <fixed_shape_tensor B> auto operator/(const B &b) const;

    template <fixed_shape_tensor Other> bool operator==(const Other &other) const {
        static_assert(compatible_for_element_wise_op<Derived, Other>(),
                      "Incompatible shapes for element-wise addition");
        auto it = static_cast<Derived *>(this)->begin();
        for (const auto &elem : other) {
            if (*it++ != elem) {
                return false;
            }
        }
        return true;
    }

    template <fixed_shape_tensor Other> bool operator!=(const Other &other) const { return !(*this == other); }
};

// Element-wise operations between fixed tensors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator+(const A &a, const B &b) {
    static_assert(compatible_for_element_wise_op<A, B>(), "Incompatible shapes for element-wise addition");
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ += elem;
    }
    return result;
}

template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator-(const A &a, const B &b) {
    static_assert(compatible_for_element_wise_op<A, B>(), "Incompatible shapes for element-wise subtraction");
    auto result = a;
    auto it = result.begin();
    for (const auto &elem : b) {
        *it++ -= elem;
    }
    return result;
}

// Matrix multiplication between fixed tensors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto operator*(const A &a, const B &b) {
    static_assert(compatible_for_matmul<A, B>(), "Incompatible shapes for matrix multiplication");
    static_assert(compatible_for_type<A, B>(), "Incompatible types for BLAS matrix multiplication");
    // matmul using BLAS
    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();

    constexpr int m = a_shape[0];
    constexpr int n = a_shape[1];
    constexpr int k = b_shape[1];

    // Determine leading dimensions based on layout
    constexpr int lda = (A::get_layout() == layout::row_major) ? a_strides[0] : a_strides[1];
    constexpr int ldb = (B::get_layout() == layout::row_major) ? b_strides[0] : b_strides[1];
    constexpr int ldc = (A::get_layout() == layout::row_major) ? a_strides[0] : a_strides[1];

    auto result = fixed_tensor<decltype(typename A::value_type{} * typename B::value_type{}), A::get_layout(),
                               A::get_error_checking(), m, k>();

    // Determine BLAS layout
    auto layout = (A::get_layout() == layout::row_major) ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;

    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        cblas_sgemm(layout, CblasNoTrans, CblasNoTrans, m, k, n, 1.0F, const_cast<float *>(a.raw_data()), lda,
                    const_cast<float *>(b.raw_data()), ldb, 0.0F, result.raw_data(), ldc);
    } else {
        cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, const_cast<double *>(a.raw_data()), lda,
                    const_cast<double *>(b.raw_data()), ldb, 0.0, result.raw_data(), ldc);
    }

    return result;
}

// Solve linear system of equations Ax = b
template <fixed_shape_tensor A, fixed_shape_tensor B> auto solve(A &a, B &b) {
    static_assert(compatible_for_solve<A, B>(), "Incompatible types or shapes for solving linear system");

    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();

    constexpr int n = a_shape[0];
    constexpr int nrhs = (B::rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    constexpr int lda = (A::get_layout() == layout::row_major) ? a_strides[0] : a_strides[1];
    constexpr int ldb = (B::get_layout() == layout::row_major) ? ((B::rank() == 1) ? 1 : b_strides[0])
                                                               : ((B::rank() == 1) ? n : b_strides[1]);

    std::vector<int> ipiv(n);

    // Determine LAPACK layout
    int lapack_layout = (A::get_layout() == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

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

// Solve linear least squares problem Ax = b
template <fixed_shape_tensor A, fixed_shape_tensor B> auto solve_lls(A &a, const B &b) {
    static_assert(A::rank() == 2, "Matrix A must be 2-dimensional");
    static_assert(B::rank() == 1 || B::rank() == 2, "B must be 1D or 2D");
    static_assert(A::constexpr_shape()[0] == B::constexpr_shape()[0],
                  "Matrix A and vector/matrix B dimensions must match");

    constexpr auto a_shape = A::constexpr_shape();
    constexpr auto b_shape = B::constexpr_shape();
    constexpr auto a_strides = A::constexpr_strides();
    constexpr auto b_strides = B::constexpr_strides();

    constexpr int m = a_shape[0];
    constexpr int n = a_shape[1];
    constexpr int nrhs = (B::rank() == 1) ? 1 : b_shape[1];

    // Determine leading dimensions based on layout
    constexpr int lda = (A::get_layout() == layout::row_major) ? a_strides[0] : a_strides[1];
    constexpr int ldb = (B::get_layout() == layout::row_major) ? ((B::rank() == 1) ? 1 : b_strides[0])
                                                               : ((B::rank() == 1) ? std::max(m, n) : b_strides[1]);

    // Determine LAPACK layout
    int lapack_layout = (A::get_layout() == layout::row_major) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

    // Create a copy of b to store the solution, ensuring it's large enough for any case
    auto x = fixed_tensor < typename B::value_type, B::get_layout(), B::get_error_checking(), std::max(m, n),
         (B::rank() == 1) ? 1 : nrhs > ::zeros();
    std::copy(b.begin(), b.end(), x.begin());

    int info;
    if constexpr (std::is_same_v<decltype(a.raw_data()), float *> ||
                  std::is_same_v<decltype(a.raw_data()), const float *>) {
        info = LAPACKE_sgels(lapack_layout, 'N', m, n, nrhs, const_cast<float *>(a.raw_data()), lda, x.raw_data(), ldb);
    } else {
        info =
            LAPACKE_dgels(lapack_layout, 'N', m, n, nrhs, const_cast<double *>(a.raw_data()), lda, x.raw_data(), ldb);
    }

    if (info != 0) {
        throw std::runtime_error("LAPACKE_gels failed with error code " + std::to_string(info));
    }

    return x;
}

// Cross product between two 3D vectors
template <fixed_shape_tensor A, fixed_shape_tensor B> auto cross(const A &a, const B &b) {
    static_assert(A::size() == 3 && B::size() == 3, "Cross product requires 3D vectors");
    using result_value_type = decltype(typename A::value_type{} * typename B::value_type{});
    auto result = fixed_tensor<result_value_type, A::get_layout(), A::get_error_checking(), 3>();
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

template <fixed_shape_tensor A, fixed_shape_tensor B, typename Epsilon>
bool approx_equal(const A &a, const B &b, const Epsilon &epsilon) {
    static_assert(compatible_for_element_wise_op<A, B>(), "Incompatible shapes for element-wise subtraction");
    auto it = a.begin();
    for (const auto &elem : b) {
        if (!approx_equal(*it++, elem, epsilon)) {
            return false;
        }
    }
    return true;
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

// Dynamic tensor with linear algebra
template <typename Derived, error_checking ErrorChecking>
class dynamic_linear_algebra_mixin : public linear_algebra_mixin<Derived, ErrorChecking> {
  public:
    template <typename Scalar> auto &operator*=(const Scalar &scalar);
    template <typename Scalar> auto &operator/=(const Scalar &scalar);
    template <tensor Other> auto &operator+=(const Other &other);
    template <tensor Other> auto &operator-=(const Other &other);
    template <tensor B> auto operator/(const B &b) const;
    template <tensor Other> bool operator==(const Other &other) const;
    template <tensor Other> bool operator!=(const Other &other) const;
};

} // namespace squint

#endif // SQUINT_LINEAR_ALGEBRA_HPP