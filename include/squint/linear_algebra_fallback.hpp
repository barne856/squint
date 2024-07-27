#ifndef SQUINT_LINEAR_ALGEBRA_FALLBACK_HPP
#define SQUINT_LINEAR_ALGEBRA_FALLBACK_HPP
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

namespace squint {

// Define necessary enums and structs for BLAS compatibility
enum class CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum class CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

// Helper functions
template <typename T> void swap_row(T *matrix, int row1, int row2, int n, int lda) {
    for (int j = 0; j < n; ++j) {
        std::swap(matrix[row1 * lda + j], matrix[row2 * lda + j]);
    }
}

template <typename T> T abs_complex(const std::complex<T> &z) { return std::abs(z); }

template <typename T> T abs_complex(const T &x) { return std::abs(x); }

// Matrix element access based on layout
template <typename T> T &matrix_element(T *matrix, int i, int j, int lda, int matrix_layout) {
    return (matrix_layout == LAPACK_ROW_MAJOR) ? matrix[i * lda + j] : matrix[j * lda + i];
}

template <typename T>
void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha, const T *a,
          int lda, const T *b, int ldb, T beta, T *c, int ldc) {
    bool row_major = (order == CBLAS_ORDER::CblasRowMajor);
    bool trans_a_bool = (trans_a != CBLAS_TRANSPOSE::CblasNoTrans);
    bool trans_b_bool = (trans_b != CBLAS_TRANSPOSE::CblasNoTrans);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = 0;
            for (int l = 0; l < k; ++l) {
                int a_idx =
                    trans_a_bool ? (row_major ? l * lda + i : i * lda + l) : (row_major ? i * lda + l : l * lda + i);
                int b_idx =
                    trans_b_bool ? (row_major ? j * ldb + l : l * ldb + j) : (row_major ? l * ldb + j : j * ldb + l);
                sum += a[a_idx] * b[b_idx];
            }
            int c_idx = row_major ? i * ldc + j : j * ldc + i;
            c[c_idx] = alpha * sum + beta * c[c_idx];
        }
    }
}

template <typename T> int getrf(int matrix_layout, int m, int n, T *a, int lda, int *ipiv) {
    if (matrix_layout != LAPACK_ROW_MAJOR && matrix_layout != LAPACK_COL_MAJOR) {
        throw std::invalid_argument("Invalid matrix layout");
    }

    int min_mn = std::min(m, n);

    for (int i = 0; i < min_mn; ++i) {
        // Find pivot
        int pivot = i;
        T max_val = abs_complex(matrix_element(a, i, i, lda, matrix_layout));

        for (int j = i + 1; j < m; ++j) {
            T val = abs_complex(matrix_element(a, j, i, lda, matrix_layout));
            if (val > max_val) {
                max_val = val;
                pivot = j;
            }
        }

        ipiv[i] = pivot + 1; // LAPACK uses 1-based indexing for ipiv

        // Swap rows if necessary
        if (pivot != i) {
            if (matrix_layout == LAPACK_ROW_MAJOR) {
                swap_row(a, i, pivot, n, lda);
            } else {
                for (int k = 0; k < m; ++k) {
                    std::swap(a[k * lda + i], a[k * lda + pivot]);
                }
            }
        }

        // Gaussian elimination
        if (matrix_element(a, i, i, lda, matrix_layout) != T(0)) {
            for (int j = i + 1; j < m; ++j) {
                T factor = matrix_element(a, j, i, lda, matrix_layout) / matrix_element(a, i, i, lda, matrix_layout);
                matrix_element(a, j, i, lda, matrix_layout) = factor;

                for (int k = i + 1; k < n; ++k) {
                    matrix_element(a, j, k, lda, matrix_layout) -= factor * matrix_element(a, i, k, lda, matrix_layout);
                }
            }
        } else if (i == min_mn - 1) {
            return i + 1; // Matrix is singular
        }
    }

    return 0; // Success
}

template <typename T> int getri(int matrix_layout, int n, T *a, int lda, const int *ipiv) {
    if (matrix_layout != LAPACK_ROW_MAJOR && matrix_layout != LAPACK_COL_MAJOR) {
        throw std::invalid_argument("Invalid matrix layout");
    }

    // Create identity matrix
    T *work = new T[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            work[i * n + j] = (i == j) ? T(1) : T(0);
        }
    }

    // Apply permutations to identity matrix
    for (int i = n - 1; i >= 0; --i) {
        int pivot = ipiv[i] - 1; // Convert back to 0-based indexing
        if (pivot != i) {
            swap_row(work, i, pivot, n, n);
        }
    }

    // Solve the system LY = P
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                work[j * n + k] -= matrix_element(a, j, i, lda, matrix_layout) * work[i * n + k];
            }
        }
    }

    // Solve the system UX = Y
    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < n; ++j) {
            work[i * n + j] /= matrix_element(a, i, i, lda, matrix_layout);
        }

        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < n; ++k) {
                work[j * n + k] -= matrix_element(a, j, i, lda, matrix_layout) * work[i * n + k];
            }
        }
    }

    // Copy result back to a
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix_element(a, i, j, lda, matrix_layout) = work[i * n + j];
        }
    }

    delete[] work;
    return 0; // Success
}

template <typename T> int gesv(int matrix_layout, int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb) {
    // Perform LU decomposition
    int info = getrf(matrix_layout, n, n, a, lda, ipiv);
    if (info != 0)
        return info;

    bool row_major = (matrix_layout == LAPACK_ROW_MAJOR);

    // Solve the system using forward and backward substitution
    for (int k = 0; k < nrhs; ++k) {
        // Forward substitution
        for (int i = 0; i < n; ++i) {
            int pivot = ipiv[i] - 1; // LAPACK uses 1-based indexing for ipiv
            if (pivot != i) {
                int idx_i = row_major ? i * ldb + k : k * ldb + i;
                int idx_pivot = row_major ? pivot * ldb + k : k * ldb + pivot;
                std::swap(b[idx_i], b[idx_pivot]);
            }
            for (int j = i + 1; j < n; ++j) {
                int idx_j = row_major ? j * ldb + k : k * ldb + j;
                int idx_i = row_major ? i * ldb + k : k * ldb + i;
                int idx_ji = row_major ? j * lda + i : i * lda + j;
                b[idx_j] -= a[idx_ji] * b[idx_i];
            }
        }

        // Backward substitution
        for (int i = n - 1; i >= 0; --i) {
            int idx_i = row_major ? i * ldb + k : k * ldb + i;
            T sum = b[idx_i];
            for (int j = i + 1; j < n; ++j) {
                int idx_j = row_major ? j * ldb + k : k * ldb + j;
                int idx_ij = row_major ? i * lda + j : j * lda + i;
                sum -= a[idx_ij] * b[idx_j];
            }
            int idx_ii = row_major ? i * lda + i : i * lda + i;
            b[idx_i] = sum / a[idx_ii];
        }
    }

    return 0;
}

template <typename T> int gels(int matrix_layout, char trans, int m, int n, int nrhs, T *a, int lda, T *b, int ldb) {
    // Input validation
    if (matrix_layout != LAPACK_ROW_MAJOR && matrix_layout != LAPACK_COL_MAJOR) {
        throw std::invalid_argument("Invalid matrix layout");
    }
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        throw std::invalid_argument("Invalid trans parameter");
    }
    if (m < 0 || n < 0 || nrhs < 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    if (lda < std::max(1, (matrix_layout == LAPACK_ROW_MAJOR) ? n : m)) {
        throw std::invalid_argument("Invalid lda");
    }
    if (ldb < std::max(1, std::max(m, n))) {
        throw std::invalid_argument("Invalid ldb");
    }

    // Determine the problem dimensions
    int nrows = (trans == 'N') ? m : n;
    int ncols = (trans == 'N') ? n : m;

    // Perform QR factorization using Householder reflections
    for (int k = 0; k < std::min(nrows, ncols); ++k) {
        // Compute the Householder vector
        T norm = 0;
        for (int i = k; i < nrows; ++i) {
            int idx = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
            norm += a[idx] * a[idx];
        }
        norm = std::sqrt(norm);

        T alpha = a[(matrix_layout == LAPACK_ROW_MAJOR) ? k * lda + k : k * lda + k];
        T beta = (alpha >= 0) ? norm : -norm;
        T tau = (beta - alpha) / beta;

        a[(matrix_layout == LAPACK_ROW_MAJOR) ? k * lda + k : k * lda + k] = beta;

        // Apply Householder reflection to A
        for (int j = k + 1; j < ncols; ++j) {
            T dot_product = 0;
            for (int i = k; i < nrows; ++i) {
                int idx_ki = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
                int idx_ji = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + j : j * lda + i;
                dot_product += a[idx_ki] * a[idx_ji];
            }
            dot_product *= tau;

            for (int i = k; i < nrows; ++i) {
                int idx_ki = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
                int idx_ji = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + j : j * lda + i;
                a[idx_ji] -= a[idx_ki] * dot_product;
            }
        }

        // Apply Householder reflection to B
        for (int j = 0; j < nrhs; ++j) {
            T dot_product = 0;
            for (int i = k; i < nrows; ++i) {
                int idx_ki = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
                int idx_ji = (matrix_layout == LAPACK_ROW_MAJOR) ? i * ldb + j : j * ldb + i;
                dot_product += a[idx_ki] * b[idx_ji];
            }
            dot_product *= tau;

            for (int i = k; i < nrows; ++i) {
                int idx_ki = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
                int idx_ji = (matrix_layout == LAPACK_ROW_MAJOR) ? i * ldb + j : j * ldb + i;
                b[idx_ji] -= a[idx_ki] * dot_product;
            }
        }
    }

    // Back-substitution
    for (int j = 0; j < nrhs; ++j) {
        for (int i = std::min(nrows, ncols) - 1; i >= 0; --i) {
            int idx_ii = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + i : i * lda + i;
            int idx_ji = (matrix_layout == LAPACK_ROW_MAJOR) ? i * ldb + j : j * ldb + i;

            for (int k = i + 1; k < ncols; ++k) {
                int idx_ki = (matrix_layout == LAPACK_ROW_MAJOR) ? i * lda + k : k * lda + i;
                int idx_jk = (matrix_layout == LAPACK_ROW_MAJOR) ? k * ldb + j : j * ldb + k;
                b[idx_ji] -= a[idx_ki] * b[idx_jk];
            }
            b[idx_ji] /= a[idx_ii];
        }
    }

    return 0; // Success
}
} // namespace squint
#endif // SQUINT_LINEAR_ALGEBRA_FALLBACK_HPP
