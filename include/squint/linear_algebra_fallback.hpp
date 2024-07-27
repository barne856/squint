#include <cmath>
#include <vector>
#include <stdexcept>

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

namespace squint {

// Define necessary enums and structs for BLAS compatibility
enum class CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum class CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

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
    bool row_major = (matrix_layout == LAPACK_ROW_MAJOR);

    for (int i = 0; i < std::min(m, n); ++i) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < m; ++j) {
            int idx_j = row_major ? j * lda + i : i * lda + j;
            int idx_pivot = row_major ? pivot * lda + i : i * lda + pivot;
            if (std::abs(a[idx_j]) > std::abs(a[idx_pivot])) {
                pivot = j;
            }
        }
        ipiv[i] = pivot + 1; // LAPACK uses 1-based indexing for ipiv

        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                int idx_i = row_major ? i * lda + j : j * lda + i;
                int idx_pivot = row_major ? pivot * lda + j : j * lda + pivot;
                std::swap(a[idx_i], a[idx_pivot]);
            }
        }

        // Gaussian elimination
        int idx_ii = row_major ? i * lda + i : i * lda + i;
        if (a[idx_ii] != 0) {
            for (int j = i + 1; j < m; ++j) {
                int idx_ji = row_major ? j * lda + i : i * lda + j;
                T factor = a[idx_ji] / a[idx_ii];
                a[idx_ji] = factor;
                for (int k = i + 1; k < n; ++k) {
                    int idx_jk = row_major ? j * lda + k : k * lda + j;
                    int idx_ik = row_major ? i * lda + k : k * lda + i;
                    a[idx_jk] -= factor * a[idx_ik];
                }
            }
        }
    }
    return 0;
}

template <typename T> int getri(int matrix_layout, int n, T *a, int lda, const int *ipiv) {
    bool row_major = (matrix_layout == LAPACK_ROW_MAJOR);
    std::vector<T> work(n);

    // Compute inverse of upper triangular matrix
    for (int i = 0; i < n; ++i) {
        int idx_ii = row_major ? i * lda + i : i * lda + i;
        a[idx_ii] = 1 / a[idx_ii];
        for (int j = 0; j < i; ++j) {
            T sum = 0;
            for (int k = j; k < i; ++k) {
                int idx_jk = row_major ? j * lda + k : k * lda + j;
                int idx_ki = row_major ? k * lda + i : i * lda + k;
                sum -= a[idx_jk] * a[idx_ki];
            }
            int idx_ji = row_major ? j * lda + i : i * lda + j;
            a[idx_ji] = sum * a[idx_ii];
        }
    }

    // Solve the system inv(A)*L = inv(U) for inv(A)
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            int idx_ij = row_major ? i * lda + j : j * lda + i;
            work[j] = a[idx_ij];
            a[idx_ij] = 0;
        }
        for (int j = i + 1; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int idx_ik = row_major ? i * lda + k : k * lda + i;
                int idx_jk = row_major ? j * lda + k : k * lda + j;
                a[idx_ik] -= work[j] * a[idx_jk];
            }
        }
    }

    // Apply row interchanges
    for (int i = n - 1; i >= 0; --i) {
        int pivot = ipiv[i] - 1; // LAPACK uses 1-based indexing for ipiv
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                int idx_i = row_major ? i * lda + j : j * lda + i;
                int idx_pivot = row_major ? pivot * lda + j : j * lda + pivot;
                std::swap(a[idx_i], a[idx_pivot]);
            }
        }
    }

    return 0;
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
    bool row_major = (matrix_layout == LAPACK_ROW_MAJOR);

    if (trans == 'N') {
        // Compute QR decomposition of A
        std::vector<T> tau(std::min(m, n));
        for (int k = 0; k < std::min(m, n); ++k) {
            // Compute Householder reflection
            T norm = 0;
            for (int i = k; i < m; ++i) {
                int idx_ik = row_major ? i * lda + k : k * lda + i;
                norm += a[idx_ik] * a[idx_ik];
            }
            norm = std::sqrt(norm);
            int idx_kk = row_major ? k * lda + k : k * lda + k;
            T alpha = a[idx_kk];
            T beta = (alpha >= 0) ? norm : -norm;
            tau[k] = (beta - alpha) / beta;
            a[idx_kk] = beta;

            // Apply Householder reflection to A and b
            for (int j = k + 1; j < n; ++j) {
                T dot = 0;
                for (int i = k; i < m; ++i) {
                    int idx_ik = row_major ? i * lda + k : k * lda + i;
                    int idx_ij = row_major ? i * lda + j : j * lda + i;
                    dot += a[idx_ik] * a[idx_ij];
                }
                dot *= tau[k];
                for (int i = k; i < m; ++i) {
                    int idx_ik = row_major ? i * lda + k : k * lda + i;
                    int idx_ij = row_major ? i * lda + j : j * lda + i;
                    a[idx_ij] -= dot * a[idx_ik];
                }
            }
            for (int j = 0; j < nrhs; ++j) {
                T dot = 0;
                for (int i = k; i < m; ++i) {
                    int idx_ik = row_major ? i * lda + k : k * lda + i;
                    int idx_ij = row_major ? i * ldb + j : j * ldb + i;
                    dot += a[idx_ik] * b[idx_ij];
                }
                dot *= tau[k];
                for (int i = k; i < m; ++i) {
                    int idx_ik = row_major ? i * lda + k : k * lda + i;
                    int idx_ij = row_major ? i * ldb + j : j * ldb + i;
                    b[idx_ij] -= dot * a[idx_ik];
                }
            }
        }

        // Solve R * x = Q^T * b
        for (int k = 0; k < nrhs; ++k) {
            for (int i = std::min(m, n) - 1; i >= 0; --i) {
                int idx_ik = row_major ? i * ldb + k : k * ldb + i;
                T sum = b[idx_ik];
                for (int j = i + 1; j < n; ++j) {
                    int idx_ij = row_major ? i * lda + j : j * lda + i;
                    int idx_jk = row_major ? j * ldb + k : k * ldb + j;
                    sum -= a[idx_ij] * b[idx_jk];
                }
                int idx_ii = row_major ? i * lda + i : i * lda + i;
                b[idx_ik] = sum / a[idx_ii];
            }
        }
    } else {
        throw std::runtime_error("Transposed case not implemented in gels_fallback");
    }

    return 0;
}
} // namespace squint