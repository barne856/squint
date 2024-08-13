# Overview


Add steps to subviews
impl linalg with BLAS backends
optimize small tensor ops with SIMD

Harder, but maybe?
broadcasting views using modulo indexing to avoid copy
einsum
fused operations at compile time with computation graph See Fastor

# List of Test Cases

1. Fixed Tensor Sizeof Test
   - Mat4

2. Fixed Tensor Creation and Basic Operations
   - Default constructor
   - Constructor with initializer list
   - Constructor with value
   - Constructor with tensor block
   - Constructor with array of blocks
   - Copy constructor
   - Move constructor
   - Assignment operator
   - Assignment from const
   - Move assignment operator

3. Fixed Tensor Element Access
   - Multidimensional subscript operator[]
   - Multidimensional subscript operator()
   - at() method
   - at() method with vector of indices

4. Fixed Tensor Layout and Strides
   - Row-major layout
   - Column-major layout

5. Fixed Tensor Views
   - Create view
   - Create const view
   - Modify through view
   - Create subview
   - Modify through subview
   - Assign from const tensor
   - Assign from const view

6. Fixed Tensor Iteration
   - Range-based for loop
   - Iterator-based loop
   - Const iteration
   - Non const iteration

7. Fixed Tensor Subview Iteration
   - Iterate over 1x4 subviews

8. Dynamic Tensor Creation and Basic Operations
   - Constructor with shape
   - Constructor with shape and layout
   - Constructor with vector of elements
   - Constructor with value
   - Constructor with tensor block
   - Constructor with array of blocks
   - Copy constructor
   - Move constructor
   - Assignment operator
   - Move assignment operator

9. Dynamic Tensor Element Access
   - Multidimensional subscript operator[]
   - Multidimensional subscript operator()
   - at() method
   - at() method with vector of indices

10. Dynamic Tensor Layout and Strides
    - Row-major layout
    - Column-major layout

11. Dynamic Tensor Views
    - Create view
    - Create const view
    - Modify through view
    - Create subview
    - Modify through subview
    - Assign from const tensor
    - Assign from const view

12. Dynamic Tensor Iteration
    - Range-based for loop
    - Iterator-based loop
    - Const iteration

13. Dynamic Tensor Subview Iteration
    - Iterate over 1x4 subviews

14. Fixed Tensor with Error Checking
    - Out of bounds access
    - Subview out of bounds

15. Dynamic Tensor with Error Checking
    - Out of bounds access
    - Invalid number of indices
    - Subview out of bounds

16. Tensor View Error Checking
    - Fixed tensor view with error checking
    - Dynamic tensor view with error checking

17. Error Checking Disabled
    - Fixed tensor without error checking
    - Dynamic tensor without error checking

18. fixed_tensor static methods
    - zeros
    - ones
    - full
    - arange
    - diag
    - diag with scalar
    - random
    - I

19. dynamic_tensor static methods
    - zeros
    - ones
    - full
    - arange
    - diag
    - diag with scalar
    - random
    - I

20. fixed_tensor fill and flatten methods
    - fill
    - flatten non-const
    - flatten const
    - view flatten
    - view flatten const

21. dynamic_tensor fill and flatten methods
    - fill
    - flatten non-const
    - flatten const
    - view flatten
    - view flatten const

22. fixed_tensor rows() and cols() tests
    - 2D tensor (rows() and cols())
    - 3D tensor (rows() and cols())
    - 1D tensor (rows() and cols())

23. dynamic_tensor rows() and cols() tests
    - 2D tensor (rows() and cols())
    - 3D tensor (rows() and cols())
    - 1D tensor (rows() and cols())

24. Diagonal Views
    - Fixed tensor 2D
    - Fixed tensor 2D const
    - Fixed tensor 2D subview
    - Fixed tensor 2D subview const
    - Dynamic tensor 2D
    - Dynamic tensor 2D const
    - Dynamic tensor 2D subview
    - Dynamic tensor 2D subview const

