#ifndef SQUINT_TENSOR_BLAS_BACKEND_HPP
#define SQUINT_TENSOR_BLAS_BACKEND_HPP

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
#define cblas_sgemm gemm<float>
#define cblas_dgemm gemm<double>
#define LAPACKE_sgetrf getrf<float>
#define LAPACKE_dgetrf getrf<double>
#define LAPACKE_sgetri getri<float>
#define LAPACKE_dgetri getri<double>
#define LAPACKE_sgesv gesv<float>
#define LAPACKE_dgesv gesv<double>
#define LAPACKE_sgels gels<float>
#define LAPACKE_dgels gels<double>
#endif

#endif // SQUINT_TENSOR_BLAS_BACKEND_HPP