
math (
    solve,
    solve_general,
    inv,
    pinv,

    cross,
    dot,
    trace,
    norm,
    squared_norm,
    mean, sum, min, max,
    approx_equal
    )

tensor_ops (/, tensor_contration)

blas_backend.hpp - just includes the blas backend with preprocessor (MKL, JIT, OpenBLAS, NONE)
blas_backend_none.hpp - a low performance fallback in pure C++ providing needed blas and lapack APIs
