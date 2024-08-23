
Usage
=====



Quantity Usage
--------------


Quantities in SQUINT can be used in a variety of ways, showcasing their flexibility and integration with both scalar and tensor operations:

.. code-block::

   // Basic quantity creation and arithmetic
   auto length = length_t<double>::meters(10.0);
   auto time = time_t<double>::seconds(2.0);
   auto velocity = length / time;
   
   // Using quantities with mathematical functions
   auto acceleration = length_t<double>::meters(9.81) / (time_t<double>::seconds(1) * time_t<double>::seconds(1));
   auto kinetic_energy = 0.5 * mass_t<double>::kilograms(2.0) * pow<2>(velocity);
   
   // Dimensionless quantities
   auto ratio = length / length_t<double>::meters(5.0);
   double scalar_value = std::sin(ratio);  // ratio can be used directly in std::sin
   
   // Quantity-aware tensors
   vec3_t<length_t<double>> position{
       length_t<double>::meters(1.0),
       length_t<double>::meters(2.0),
       length_t<double>::meters(3.0)
   };
   
   // Mixing quantities and scalars in tensor operations
   auto scaled_position = position * 2.0;  // Results in a vec3_t<length_t<double>>


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
       length_t<double>::meters(1.0),
       length_t<double>::meters(2.0),
       length_t<double>::meters(3.0)
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


Tensor Contraction (for dynamic tensors)
----------------------------------------


- **Tensor Contraction**:

.. code-block::

   auto contracted = contract(A, B, contraction_pairs);

For tensors :math:`A` and :math:`B`, the contraction over indices :math:`i` and :math:`j` is:
  
:math:`(A \cdot B)_{k_1...k_n l_1...l_m} = \sum_{i,j} A_{k_1...k_n i} B_{j l_1...l_m}`

