Tensors
=======

Tensor Representation and Order
-------------------------------

Column-Centric Approach
^^^^^^^^^^^^^^^^^^^^^^^

SQUINT adopts a column-centric approach to tensor representation, which provides a theoretically useful perspective, particularly in the context of linear algebra and multidimensional data analysis. This approach is distinct from the traditional row-major or column-major memory layouts and instead focuses on the conceptual structure of tensors.

Representation
^^^^^^^^^^^^^^

In SQUINT's column-centric approach:

1. A 1D tensor (vector) is represented as a column with shape m
2. A 2D tensor (matrix) has shape m × n (a single row being of shape 1 × n)
3. A 3D tensor has shape m × n × l
4. Higher-order tensors follow this pattern, adding dimensions to the right

This means that indexing a tensor is done by specifying the row index first, followed by the column index, and so on for higher-order tensors. The shape of a tensor is defined by the number of elements in each dimension, starting from the leftmost dimension.

Examples:

.. math::

   \text{1D tensor (vector):} \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} \text{shape: } m

   \text{2D tensor (matrix):} \begin{bmatrix} 
   a_{11} & a_{12} & \cdots & a_{1n} \\
   a_{21} & a_{22} & \cdots & a_{2n} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{m1} & a_{m2} & \cdots & a_{mn}
   \end{bmatrix} \text{shape: } m \times n

   \text{3D tensor:} \text{shape: } m \times n \times l

For a 3D tensor with shape 2 × 3 × 4, it has 2 elements along the first dimension, 3 elements along the second dimension, and 4 elements along the third dimension.

Indexing:
- 1D tensor: t[i] where i is the row index
- 2D tensor: t[i, j] where i is the row index and j is the column index
- 3D tensor: t[i, j, k] where i is the row index, j is the column index, and k is the depth index

Comparison with Row-Major and Column-Major Orders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's important to distinguish between SQUINT's column-centric approach and the traditional row-major and column-major memory layouts:

1. Row-major order: Elements are stored contiguously by row.
   - 1D: [a b c]
   - 2D: [a b c | d e f]
   - 3D: [a b c | d e f || g h i | j k l]

2. Column-major order: Elements are stored contiguously by column.
   - 1D: [a b c]
   - 2D: [a d | b e | c f]
   - 3D: [a g | d j || b h | e k || c i | f l]

3. SQUINT's column-centric approach:
   - Conceptual representation, not a memory layout
   - 1D: column vector [a; b; c]
   - 2D: matrix [a b c; d e f]
   - 3D: tensor with shape m × n × l

For 1D and 2D tensors, SQUINT's approach aligns with both row-major and column-major orders in terms of indexing. The difference becomes apparent for higher-order tensors, where SQUINT's approach provides a more intuitive extension of the matrix concept.

Theoretical Usefulness
^^^^^^^^^^^^^^^^^^^^^^

This representation aligns closely with fundamental concepts in linear algebra and offers several advantages:

1. **Vector-Centric View**: By treating 1D tensors as column vectors by default, this approach emphasizes the column space of matrices, which is crucial in many linear algebra applications.

2. **Natural Extension**: The progression from vectors to matrices to higher-order tensors feels natural, with each step adding a new dimension to the right.

3. **Consistency with Mathematical Notation**: This representation aligns well with standard mathematical notation in linear algebra, where vectors are often implicitly treated as column vectors.

4. **Intuitive for Linear Transformations**: When thinking about linear transformations, it's often helpful to consider how each column of a matrix transforms a basis vector of the input space.

5. **Simplifies Certain Operations**: Operations like matrix-vector multiplication become more intuitive when visualizing the matrix columns as the fundamental building blocks.

Bridging Theory and Practice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach bridges the gap between theoretical linear algebra and practical implementation. It allows users to think about tensors in a way that's consistent with mathematical theory while still benefiting from efficient memory layouts and operations.

By adopting this column-centric view, SQUINT encourages users to think about tensors in a way that aligns with important linear algebra concepts, potentially leading to more intuitive algorithm design and better understanding of multidimensional data structures.

Tensor Construction
-------------------


SQUINT provides several ways to construct tensors, with a default column-major layout:

1. Using initializer lists (column-major order):

.. code-block::

   mat2x3 A{1, 4, 2, 5, 3, 6};
   // A represents:
   // [1 2 3]
   // [4 5 6]

2. Factory methods:

.. code-block::

   auto zero_matrix = mat3::zeros();
   auto ones_matrix = mat4::ones();
   auto identity_matrix = mat3::eye();
   auto random_matrix = mat3::random(0.0, 1.0);
   // and more ...

3. Element-wise initialization:

.. code-block::

   mat3 custom_matrix;
   for (size_t i = 0; i < 3; ++i) {
       for (size_t j = 0; j < 3; ++j) {
           custom_matrix(i, j) = i * 3 + j;  // Note the use of () for element access
       }
   }

4. Construction from other tensors or views:

.. code-block::

   mat3 original{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
   mat3 copy(original);
   
   mat4 big_matrix = mat4::random(0.0, 1.0);
   mat3 sub_matrix(big_matrix.subview<3, 3>(0, 0));

5. Dynamic tensor construction:

.. code-block::

   std::vector<size_t> shape = {3, 4, 5};
   dynamic_tensor<float> dynamic_tensor(shape);
   dynamic_tensor<float> filled_tensor(shape, 1.0f);

6. Tensor construction with quantities:

.. code-block::

   vec3_t<length_t<double>> position{
       units::meters(1.0),
       units::meters(2.0),
       units::meters(3.0)
   };


Basic Operations
----------------


SQUINT supports a wide range of operations for tensors:

.. code-block::

   auto C = A + B;  // Element-wise addition
   auto D = A * B;  // Matrix multiplication
   auto E = A * 2.0;  // Scalar multiplication
   auto F = A / B;  // General least squares / least norm solution
   
   // Element access (note the use of () for multi-dimensional access)
   auto element = A(1, 2);  // Access element at row 1, column 2
   
   // Iteration (column-major order by default)
   for (const auto& element : A) {
       // Process each element
   }
   
   // Iteration of rows
   for (const auto& row : A.rows()) {
       // Process each row
   }
   
   // Iteration of views
   for (const auto& view : A.subviews<2,3>()) {
       // Process each view
   }

For matrix multiplication, the operation performed is:

:math:`(AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj}`


Views and Reshaping
-------------------


SQUINT provides powerful view and reshaping capabilities:

.. code-block::

   auto view = A.view();  // Create a view of the entire tensor
   auto subview = A.subview<2, 2>(0, 1);  // Create a 2x2 subview starting at (0, 1)
   auto reshaped = A.reshape<6>();  // Reshape to a 1
   D tensor
   auto transposed = A.transpose();  // Transpose the tensor
   auto permuted = A.permute<1,0>(); // Permutation of the tensor
   
   // For dynamic tensors
   auto dynamic_reshaped = dynamic_tensor.reshape({6, 4});
   auto dynamic_transposed = dynamic_tensor.transpose();


Linear Algebra Operations
-------------------------


SQUINT provides comprehensive linear algebra operations:

- **Solving Linear Systems**:

.. code-block::

   auto result = solve(A, b);  // Solves Ax = b for square systems

This solves the system of linear equations:
  
:math:`Ax = b`

A will be overwritten with the LU decomposition of A and b will be overwritten with the solution x.

- **Least Squares / Least Norm Solution**:

.. code-block::

   auto result = solve_general(A, b);  // Solves Ax = b for non-square systems

:math:`Ax = b`

The system is solved in the least squares sense, where A is an m x n matrix with m >= n and in the least norm sense when m < n.

A will be overwritten with the QR decomposition of A and b will be overwritten with the solution x.

.. note::
   b must have enough rows to store the solution.

- **Matrix Inversion**:

.. code-block::

   auto inverse = inv(A);  // Computes the inverse of a square matrix

The inverse :math:`A^{-1}` satisfies:
  
:math:`AA^{-1} = A^{-1}A = I`

- **Pseudoinverse**:

.. code-block::

   auto pseudo_inverse = pinv(A);  // Computes the Moore-Penrose pseudoinverse

For a matrix :math:`A`, the Moore-Penrose pseudoinverse :math:`A^+` satisfies:
  
:math:`AA^+A = A`
:math:`A^+AA^+ = A^+`
:math:`(AA^+)^* = AA^+`
:math:`(A^+A)^* = A^+A`


Vector Operations
-----------------


- **Cross Product** (for 3D vectors):

.. code-block::

   auto cross_product = cross(a, b);

For vectors :math:`a = (a_x, a_y, a_z)` and :math:`b = (b_x, b_y, b_z)`:
  
:math:`a \times b = (a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x)`

- **Dot Product**:

.. code-block::

   auto dot_product = dot(a, b);

For vectors :math:`a` and :math:`b`:
  
:math:`a \cdot b = \sum_{i=1}^n a_i b_i`

- **Vector Norm**:

.. code-block::

   auto vector_norm = norm(a);

The Euclidean norm of a vector :math:`a` is:
  
:math:`\|a\| = \sqrt{\sum_{i=1}^n |a_i|^2}`


Matrix Operations
-----------------


- **Trace**:

.. code-block::

   auto matrix_trace = trace(A);

The trace of a square matrix :math:`A` is:
  
:math:`\text{tr}(A) = \sum_{i=1}^n A_{ii}`


Statistical Functions
---------------------


- **Mean**:

.. code-block::

   auto tensor_mean = mean(A);

For a tensor :math:`A` with :math:`n` elements:
  
:math:`\text{mean}(A) = \frac{1}{n} \sum_{i=1}^n A_i`


Tensor Contraction
------------------


- **Tensor Contraction**:

.. code-block::

   auto contracted = contract(A, B, contraction_pairs);

For tensors :math:`A` and :math:`B`, the contraction over indices :math:`i` and :math:`j` is:
  
:math:`(A \cdot B)_{k_1...k_n l_1...l_m} = \sum_{i,j} A_{k_1...k_n i} B_{j l_1...l_m}`


Tensor Error Checking
---------------------

SQUINT provides optional error checking for tensors, which is separate from and orthogonal to error checking for quantities. When enabled, tensor error checking primarily focuses on bounds checking and additional shape checks at runtime, especially for dynamic tensors.

Enabling Error Checking
^^^^^^^^^^^^^^^^^^^^^^^

Error checking for tensors can be enabled by specifying the `error_checking::enabled` policy when declaring a tensor:

.. code-block:: cpp

   using ErrorTensor = squint::tensor<float, dynamic, dynamic, error_checking::enabled>
   ErrorTensor t({2,3}, std::vector<float>{1, 4, 2, 5, 3, 6});

Types of Checks
^^^^^^^^^^^^^^^

When error checking is enabled for tensors, SQUINT performs the following types of checks:

1. **Bounds Checking**: Ensures that element access is within the tensor's dimensions.

   .. code-block:: cpp

      // This will throw std::out_of_range
      t(2, 0);
      t(0, 3);

2. **Shape Compatibility**: Verifies that tensor operations are performed on compatible shapes.

   .. code-block:: cpp

      ErrorTensor a({2, 3});
      ErrorTensor b({3, 4});
      ErrorTensor c({2, 4});
      
      // This will compile and run correctly
      auto result1 = a * b;
      
      // This will throw a runtime error due to incompatible shapes
      auto result2 = a * c;

3. **View Bounds**: Ensures that tensor views and reshaping operations are within bounds.

   .. code-block:: cpp

      // This will throw if the subview exceeds the tensor's bounds
      auto subview = t.subview({2,2}, {1, 2});

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

While error checking provides additional safety, it does come with a performance cost. In performance-critical code, you may want to disable error checking:

.. code-block:: cpp

   using FastTensor = squint::tensor<float, squint::shape<2, 3>, squint::strides::column_major<squint::shape<2, 3>>, squint::error_checking::disabled>;
   FastTensor ft{1, 4, 2, 5, 3, 6};

   // No bounds checking performed, may lead to undefined behavior if accessed out of bounds
   auto element = ft(1, 1);

Error Checking and Quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's important to note that tensor error checking is independent of quantity error checking. You can have tensors of quantities with different error checking policies:

.. code-block:: cpp

   // Tensor with error checking, containing quantities without error checking
   tensor<length_t<double>, shape<3>, strides::column_major<shape<3>>, error_checking::enabled> t1;

   // Tensor without error checking, containing quantities with error checking
   tensor<quantity<double, dimensions::L, error_checking::enabled>, shape<3>, strides::column_major<shape<3>>, error_checking::disabled> t2;
