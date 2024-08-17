/**
 * @file blas_backend.hpp
 * @brief Defines the backend for BLAS and LAPACK operations.
 *
 * This file provides the backend for BLAS and LAPACK operations in the Squint library.
 * The backend is selected at compile time by defining one of the following macros:
 * - SQUINT_BLAS_BACKEND_MKL: Use Intel MKL as the backend.
 * - SQUINT_BLAS_BACKEND_OPENBLAS: Use OpenBLAS as the backend.
 * - SQUINT_BLAS_BACKEND_NONE: Use a fallback backend that provides basic implementations.
 */
#ifndef SQUINT_TENSOR_BLAS_BACKEND_HPP
#define SQUINT_TENSOR_BLAS_BACKEND_HPP
// NOLINTBEGIN
#ifdef SQUINT_BLAS_BACKEND_MKL
#define MKL_DIRECT_CALL_JIT
#include <mkl.h>
#define BLAS_INT MKL_INT
#elif defined(SQUINT_BLAS_BACKEND_OPENBLAS)
#include <cblas.h>
#include <lapacke.h>
#define BLAS_INT int
#elif defined(SQUINT_BLAS_BACKEND_NONE)
#include "squint/tensor/blas_backend_none.hpp"
#define BLAS_INT int
#define cblas_sgemm squint::gemm<float>
#define cblas_dgemm squint::gemm<double>
#define LAPACKE_sgetrf squint::getrf<float>
#define LAPACKE_dgetrf squint::getrf<double>
#define LAPACKE_sgetri squint::getri<float>
#define LAPACKE_dgetri squint::getri<double>
#define LAPACKE_sgesv squint::gesv<float>
#define LAPACKE_dgesv squint::gesv<double>
#define LAPACKE_sgels squint::gels<float>
#define LAPACKE_dgels squint::gels<double>
#define CBLAS_TRANSPOSE squint::CBLAS_TRANSPOSE
#define CBLAS_ORDER squint::CBLAS_ORDER
#define LAPACK_COL_MAJOR squint::LAPACK_COL_MAJOR
#define LAPACK_ROW_MAJOR squint::LAPACK_ROW_MAJOR
#endif
// NOLINTEND
#endif // SQUINT_TENSOR_BLAS_BACKEND_HPP