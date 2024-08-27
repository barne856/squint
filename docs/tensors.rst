Tensors
=======

Tensor Representation and Order
-------------------------------

Column-Centric Approach
^^^^^^^^^^^^^^^^^^^^^^^

SQUINT uses a unique approach to represent n-dimensional data. From here on we'll refer to this approach as *column-centric*.
Column-centric is opposed to the more common *row-centric* approach used in almost every other language that deals with multidimensional data.
In the opinion of the author, even though this may at first feel unnecessarily counter to the established conventions, the column-centric approach is more
intuitive when viewed from a purley mathematical perspective and leads to simpler implementations of mathematical and physical concepts in many cases.

.. note::
   It is important to note that the column-centric and row-centric approaches are orthogonal concepts from memory layouts. In fact, there are
   analogous memory layouts in both approaches to the traditional memory layouts *row-major* and *column-major*.

Representation
^^^^^^^^^^^^^^

In SQUINT's column-centric approach:

1. A 1st order tensor (vector) is represented as a column with shape m
2. A 2nd order tensor (matrix) has shape m × n (a single row being of shape 1 × n)
3. A 3rd order tensor has shape m × n × l
4. Higher-order tensors follow this pattern, adding dimensions to the right

This means that indexing a tensor is done by specifying the row index first, followed by the column index, and so on for higher-order tensors.
The shape of a tensor is defined by the number of elements in each dimension, starting from the leftmost dimension.

.. math::
   \begin{array}{|c|c|c|}
   \hline
   \text{Tensor Order} & \text{Representation} & \text{Shape} \\
   \hline
   \text{1st order (vector)} &
   \begin{bmatrix}
    a_0 \\ a_1 \\ \vdots \\ a_{m-1}
   \end{bmatrix} &
    m \\
   \hline
   \text{2nd order (matrix)} &
   \begin{bmatrix}
   a_{00} & a_{01} & \cdots & a_{0,n-1} \\
   a_{10} & a_{11} & \cdots & a_{1,n-1} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{m-1,0} & a_{m-1,1} & \cdots & a_{m-1,n-1}
   \end{bmatrix} &
    m \times n \\
   \hline
   \text{3rd order} &
   \begin{bmatrix}
   \begin{bmatrix}
   a_{000} & a_{010} & \cdots & a_{0,n-1,0} \\
   a_{100} & a_{110} & \cdots & a_{1,n-1,0} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{m-1,0,0} & a_{m-1,1,0} & \cdots & a_{m-1,n-1,0}
   \end{bmatrix} \\
   \begin{bmatrix}
   a_{001} & a_{011} & \cdots & a_{0,n-1,1} \\
   a_{101} & a_{111} & \cdots & a_{1,n-1,1} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{m-1,0,1} & a_{m-1,1,1} & \cdots & a_{m-1,n-1,1}
   \end{bmatrix} \\
   \vdots \\
   \begin{bmatrix}
   a_{0,0,l-1} & a_{0,1,l-1} & \cdots & a_{0,n-1,l-1} \\
   a_{1,0,l-1} & a_{1,1,l-1} & \cdots & a_{1,n-1,l-1} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{m-1,0,l-1} & a_{m-1,1,l-1} & \cdots & a_{m-1,n-1,l-1}
   \end{bmatrix}
   \end{bmatrix}
    &
    m \times n \times l \\
   \hline
   \end{array}

Indexing
^^^^^^^^^

- 1st order tensor: `t[i]` where `i` is the row index
- 2nd order tensor: `t[i, j]` where `i` is the row index and `j` is the column index
- 3rd order tensor: `t[i, j, k]` where `i` is the row index, `j` is the column index, and `k` is the depth index

Comparison with Row-Centric Approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's important to understand how SQUINT's column-centric approach differs from traditional row-centric approaches. The column-centric approach has memory layouts that are analogous to the row-centric layouts but with a different ordering of dimensions:

Consider a simple example with the following tensors and their representations below:

.. math::
  \begin{equation}
      \begin{bmatrix} a \\ b \end{bmatrix}\tag{A}
  \end{equation}

.. math::
  \begin{equation}
      \begin{bmatrix} a & c \\ b & d \end{bmatrix}\tag{B}
  \end{equation}

.. math::
  \begin{equation}
      \begin{bmatrix}\begin{bmatrix} a & c \\ b & d \end{bmatrix} \\ \begin{bmatrix} e & g \\ f & h \end{bmatrix}\end{bmatrix}\tag{C}
  \end{equation}

.. list-table:: Tensor Memory Layout Comparison
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Tensor
     - Row-Centric Row-Major
     - Row-Centric Column-Major
     - Column-Centric Row-Major
     - Column-Centric Column-Major
   * - **A**
     - | Memory: [a, b]
       | Strides: [1, 1]
       | Shape: [m, 1]
       | Indexing: t[row, column]
     - | Memory: [a, b]
       | Strides: [1, 1]
       | Shape: [m, 1]
       | Indexing: t[row, column]
     - | Memory: [a, b]
       | Strides: [1]
       | Shape: [m]
       | Indexing: t[row]
     - | Memory: [a, b]
       | Strides: [1]
       | Shape: [m]
       | Indexing: t[row]
   * - **B**
     - | Memory: [a, c, b, d]
       | Strides: [n, 1]
       | Shape: [m, n]
       | Indexing: t[row, column]
     - | Memory: [a, b, c, d]
       | Strides: [1, m]
       | Shape: [m, n]
       | Indexing: t[row, column]
     - | Memory: [a, c, b, d]
       | Strides: [n, 1]
       | Shape: [m, n]
       | Indexing: t[row, column]
     - | Memory: [a, b, c, d]
       | Strides: [1, m]
       | Shape: [m, n]
       | Indexing: t[row, column]
   * - **C**
     - | Memory: [a, c, b, d, e, f, g, h]
       | Strides: [m*n, n, 1]
       | Shape: [l, m, n]
       | Indexing: t[depth, row, column]
     - | Memory: [a, e, b, f, c, g, d, h]
       | Strides: [1, m, m*n]
       | Shape: [l, m, n]
       | Indexing: t[depth, row, column]
     - | Memory: [a, e, c, h, b, f, d, g]
       | Strides: [m*n, n, 1]
       | Shape: [m, n, l]
       | Indexing: t[row, column, depth]
     - | Memory: [a, b, c, d, e, f, g, h]
       | Strides: [1, m, m*n]
       | Shape: [m, n, l]
       | Indexing: t[row, column, depth]

.. note::
   In the 1D case (A), all representations are equivalent with the only difference being an additional dimesion of 1 for the columns needs to be added to the shape of the row-centric views in order to represent a column.
   In the 2D case (B), the column-major and row-major layout are equivalent to their corresponding layout in the other approach.
   In the 3D case (C), the column-centric approach adds the new dimension to the right, while the row-centric approach adds the new dimension to the left. This difference is reflected in the memory layout and indexing.

SQUINT uses the column-centric approach with the column-major layout by default since this is the most straightforward view when we maintain the idea of columns as the fundamental building blocks of tensors.
You can specify any sequence to represent the strides and shape of the tensor, which allows you to use any approach with any memory layout you prefer.

Tensor Construction
-------------------


SQUINT provides several ways to construct tensors:

1. Using initializer lists:

.. code-block:: cpp

   // Alias types use the default column-centric approach and column-major layout
   mat2x3 A{1, 4, 2, 5, 3, 6};
   // A represents:
   // [1 2 3]
   // [4 5 6]

2. Factory methods:

.. code-block:: cpp

   auto zero_matrix = mat3::zeros();
   auto ones_matrix = mat4::ones();
   auto identity_matrix = mat3::eye();
   auto random_matrix = mat3::random(0.0, 1.0);
   // and more ...

3. Element-wise initialization:

.. code-block:: cpp

   mat3 custom_matrix;
   for (size_t i = 0; i < 3; ++i) {
       for (size_t j = 0; j < 3; ++j) {
           custom_matrix(i, j) = i * 3 + j;  // Note the use of () for element access
       }
   }

4. Construction from other tensors or views:

.. code-block:: cpp

   mat3 original{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
   mat3 copy(original);
   
   mat4 big_matrix = mat4::random(0.0, 1.0);
   mat3 sub_matrix(big_matrix.subview<3, 3>(0, 0));

5. Dynamic tensor construction:

.. code-block:: cpp

   std::vector<size_t> shape = {3, 4, 5};
   dynamic_tensor<float> dynamic_tensor(shape);
   dynamic_tensor<float> filled_tensor(shape, 1.0f);

6. Tensor construction with quantities:

.. code-block:: cpp

   vec3_t<length_t<double>> position{
       units::meters(1.0),
       units::meters(2.0),
       units::meters(3.0)
   };


Basic Operations
----------------


SQUINT supports a wide range of operations for tensors:

.. code-block:: cpp

   auto C = A + B;  // Element-wise addition
   auto D = A * B;  // Matrix multiplication
   auto E = A * 2.0;  // Scalar multiplication
   auto F = A / B;  // General least squares or least norm solution
   
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

.. code-block:: cpp

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

.. code-block:: cpp

   auto result = solve(A, b);  // Solves Ax = b for square systems

This solves the system of linear equations:
  
:math:`Ax = b`

A will be overwritten with the LU decomposition of A and b will be overwritten with the solution x.

- **Least Squares / Least Norm Solution**:

.. code-block:: cpp

   auto result = solve_general(A, b);  // Solves Ax = b for non-square systems

:math:`Ax = b`

The system is solved in the least squares sense, where A is an m x n matrix with m >= n and in the least norm sense when m < n.

A will be overwritten with the QR decomposition of A and b will be overwritten with the solution x.

.. note::
   b must have enough rows to store the solution.

- **Matrix Inversion**:

.. code-block:: cpp

   auto inverse = inv(A);  // Computes the inverse of a square matrix

The inverse :math:`A^{-1}` satisfies:
  
:math:`AA^{-1} = A^{-1}A = I`

- **Pseudoinverse**:

.. code-block:: cpp

   auto pseudo_inverse = pinv(A);  // Computes the Moore-Penrose pseudoinverse

For a matrix :math:`A`, the Moore-Penrose pseudoinverse :math:`A^+` satisfies:
  
:math:`AA^+A = A`
:math:`A^+AA^+ = A^+`
:math:`(AA^+)^* = AA^+`
:math:`(A^+A)^* = A^+A`


Vector Operations
-----------------


- **Cross Product** (for 3D vectors):

.. code-block:: cpp

   auto cross_product = cross(a, b);

For vectors :math:`a = (a_x, a_y, a_z)` and :math:`b = (b_x, b_y, b_z)`:
  
:math:`a \times b = (a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x)`

- **Dot Product**:

.. code-block:: cpp

   auto dot_product = dot(a, b);

For vectors :math:`a` and :math:`b`:
  
:math:`a \cdot b = \sum_{i=1}^n a_i b_i`

- **Vector Norm**:

.. code-block:: cpp

   auto vector_norm = norm(a);

The Euclidean norm of a vector :math:`a` is:
  
:math:`\|a\| = \sqrt{\sum_{i=1}^n |a_i|^2}`


Matrix Operations
-----------------


- **Trace**:

.. code-block:: cpp

   auto matrix_trace = trace(A);

The trace of a square matrix :math:`A` is:
  
:math:`\text{tr}(A) = \sum_{i=1}^n A_{ii}`


Statistical Functions
---------------------


- **Mean**:

.. code-block:: cpp

   auto tensor_mean = mean(A);

For a tensor :math:`A` with :math:`n` elements:
  
:math:`\text{mean}(A) = \frac{1}{n} \sum_{i=1}^n A_i`


Tensor Contraction
------------------


- **Tensor Contraction**:

.. code-block:: cpp

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
