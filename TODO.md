element_wise_ops (+=, -=, +, -, ==, unary -)
scalar_ops (*=, /=, *, /)
tensor_ops (*, /, tensor_contration?)
math (cross, dot, solve, solve_general, inv, pinv, trace, norm, squared_norm, mean, sum, min, max)

blas_backend.hpp - just includes the blas backend with preprocessor (MKL, JIT, OpenBLAS, NONE)
blas_backend_none.hpp - a low performance fallback in pure C++ providing needed blas and lapack APIs
