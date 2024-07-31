# Overview

1. tensor_base<T, Derived>
   - Common base class for all tensor types
   - Methods:
     * rank()
     * size()
     * shape()
     * strides()
     * get_layout()
     * at(indices...)
     * operator()
     * operator[]
     * data()
     * raw_data()

2. iterable_tensor<Derived, T> : public tensor_base<T, Derived>
   - Mixin class for iteration capabilities
   - Methods:
     * begin()
     * end()
     * cbegin()
     * cend()

3. linear_algebra_mixin<Derived, T>
   - Mixin class for linear algebra operations
   - Methods:
     * operator+=, operator-=
     * operator*=, operator/= (scalar)
     * norm(), squared_norm()
     * trace()
     * mean(), sum()
     * min(), max()

4. tensor<T, Shape, Layout, ErrorChecking>
   : public tensor_base<T, tensor<T,Shape,Layout,ErrorChecking>>
   : public iterable_tensor<tensor<T,Shape,Layout,ErrorChecking>, T>
   : public linear_algebra_mixin<tensor<T,Shape,Layout,ErrorChecking>, T>
   - Unified tensor class for both fixed and dynamic tensors
   - Shape can be either std::array<std::size_t, N> for fixed or std::vector<std::size_t> for dynamic
   - Methods:
     * Constructors (default, shape, initializer list, etc.)
     * reshape()
     * transpose()
     * subview()
     * flatten()
     * diag_view()
     * rows(), cols()
     * row(index), col(index)
     
   - Static methods:
     * zeros(), ones(), full()
     * arange(), linspace()
     * random()
     * eye(), identity()

5. tensor_view<T, TensorType, Constness>
   : public tensor_base<T, tensor_view<T,TensorType,Constness>>
   : public iterable_tensor<tensor_view<T,TensorType,Constness>, T>
   : public linear_algebra_mixin<tensor_view<T,TensorType,Constness>, T>
   - Unified view class for both fixed and dynamic tensors
   - Methods:
     * Constructors
     * subview()
     * reshape()
     * transpose()
     * flatten()

6. blas_lapack_interface<Backend>
   - Template interface for BLAS/LAPACK operations
   - Specializations for different backends (OpenBLAS, MKL, fallback)
   - Methods:
     * gemm()
     * gesv()
     * getrf()
     * getri()
     * gels()

7. linear_algebra_operations
   - Free functions for linear algebra operations
   - Functions:
     * operator+, operator- (element-wise)
     * operator* (matrix multiplication)
     * operator/ (matrix division / solve)
     * solve()
     * solve_lls()
     * inv()
     * pinv()
     * dot()
     * cross()
     * normalize()

8. dimension<L, T, M, K, I, N, J>
   - Dimension class (unchanged)

9. quantity<T, D, ErrorChecking>
   - Quantity class (mostly unchanged)
   - Consider integrating error checking into the class itself

10. units namespace
    - Unit definitions and conversions (unchanged)

11. constants namespace
    - Physical and mathematical constants (unchanged)
