#ifndef SQUINT_LINEAR_ALGEBRA_HPP
#define SQUINT_LINEAR_ALGEBRA_HPP

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

    auto mean() const { return sum() / static_cast<Derived const *>(this)->size(); }
    auto sum() const {
        auto result = typename Derived::value_type{};
        for (const auto &elem : *static_cast<Derived const *>(this)) {
            result += elem;
        }
        return result;
    }
    auto min() const {
        auto result = *static_cast<Derived const *>(this)->begin();
        for (const auto &elem : *static_cast<Derived const *>(this)) {
            if (elem < result) {
                result = elem;
            }
        }
        return result;
    }
    auto max() const {
        auto result = *static_cast<Derived const *>(this)->begin();
        for (const auto &elem : *static_cast<Derived const *>(this)) {
            if (elem > result) {
                result = elem;
            }
        }
        return result;
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

    template <fixed_shape_tensor B> auto solve(const B &b) const {
        // PA=LU factorization (Doolittle algorithm)
        constexpr std::size_t count = Derived::constexpr_shape()[0];
        auto U = static_cast<Derived const *>(this)->template as<typename Derived::value_type>();
        using L_type = decltype(typename Derived::value_type{} / typename Derived::value_type{});
        auto L = static_cast<Derived const *>(this)->template as<L_type>();
        auto P = L;
        for (std::size_t i = 0; i < count; ++i) {
            P[i, i] = L_type{1};
        }
        for (std::size_t k = 0; k < count - 1; ++k) {
            // select index >= k to maximize |A[i][k]|
            std::size_t index = k;
            auto val = typename Derived::value_type{};
            for (std::size_t i = k; i < count; ++i) {
                auto a_val = U[i, k];
                a_val = a_val < typename Derived::value_type(0) ? -a_val : a_val;
                if (a_val > val) {
                    index = i;
                    val = a_val;
                }
            }
            // Swap Rows
            auto U_row = U.row(k);
            U.row(k) = U.row(index);
            U.row(index) = U_row;
            auto L_row = L.row(k);
            L.row(k) = L.row(index);
            L.row(index) = L_row;
            auto P_row = P.row(k);
            P.row(k) = P.row(index);
            P.row(index) = P_row;
            // compute factorization
            for (std::size_t j = k + 1; j < count; ++j) {
                L[j, k] = U[j, k] / U[k, k];
                for (std::size_t i = k; i < count; ++i) {
                    U[j, i] = U[j, i] - L[j, k] * U[k, i];
                }
            }
        }
        // fill diagonals of L with 1
        for (std::size_t i = 0; i < count; ++i) {
            L[i, i] = L_type{1};
        }
        // for each column in B, solve the system using forward and back substitution
        using result_type = decltype(typename Derived::value_type{} * typename B::value_type{});
        auto result = b.template as<result_type>();
        for (auto &col : result.cols()) {
            // forward substitute
            auto y = col.template as<result_type>().template reshape<count>;
            auto b_col = col.template as<result_type>().template reshape<count>;
            // permute b
            b_col = P * b_col;
            for (std::size_t i = 0; i < count; ++i) {
                auto tmp = b_col[i];
                for (std::size_t j = 0; j < i; ++j) {
                    tmp -= L[i, j] * y[j];
                }
                y[i] = tmp / L[i, i];
            }
            // back substitute into column
            for (int i = count - 1; i > -1; --i) {
                auto tmp = y[i];
                for (std::size_t j = i + 1; j < count; ++j) {
                    tmp -= U[i, j] * col[j];
                }
                col[i] = tmp / U[i, i];
            }
        }
        return result;
    }

    template <fixed_shape_tensor B> auto solve_lls(const B &b) const;

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
    constexpr auto other_rank = B::rank();
    constexpr auto this_shape = A::constexpr_shape();
    constexpr auto other_shape = B::constexpr_shape();
    // get result value_type
    using result_value_type = decltype(typename A::value_type{} * typename B::value_type{});
    if constexpr (other_rank == 2) {
        auto result =
            fixed_tensor<result_value_type, A::get_layout(), A::get_error_checking(), this_shape[0], other_shape[1]>();
        for (std::size_t i = 0; i < this_shape[0]; ++i) {
            for (std::size_t j = 0; j < other_shape[1]; ++j) {
                for (std::size_t k = 0; k < this_shape[1]; ++k) {
                    result[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return result;
    } else {
        auto result = fixed_tensor<result_value_type, A::get_layout(), A::get_error_checking(), this_shape[0]>();
        for (std::size_t i = 0; i < this_shape[0]; ++i) {
            for (std::size_t j = 0; j < this_shape[1]; ++j) {
                result[i] += a[i, j] * b[j];
            }
        }
        return result;
    }
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

    template <tensor B> auto solve(const B &b) const;

    template <tensor B> auto solve_lls(const B &b) const;

    template <tensor B> auto operator/(const B &b) const;

    template <tensor Other> bool operator==(const Other &other) const;

    template <tensor Other> bool operator!=(const Other &other) const;
};

} // namespace squint

#endif // SQUINT_LINEAR_ALGEBRA_HPP