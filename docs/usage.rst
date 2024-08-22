
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


SQUINT provides a comprehensive set of linear algebra operations:

- **Solving Linear Systems**:
  ```cpp
  auto result = solve(A, b);  // Solves Ax = b for square systems
  auto result_general = solve_general(A, b);  // Solves Ax = b for general (overdetermined or underdetermined) systems
  ```

- **Matrix Inversion**:
  ```cpp
  auto inverse = inv(A);  // Computes the inverse of a square matrix
  ```

- **Pseudoinverse**:
  ```cpp
  auto pseudo_inverse = pinv(A);  // Computes the Moore-Penrose pseudoinverse
  ```


Vector Operations
-----------------


SQUINT supports various vector operations:

- **Cross Product** (for 3D vectors):
  ```cpp
  auto cross_product = cross(a, b);
  ```

- **Dot Product**:
  ```cpp
  auto dot_product = dot(a, b);
  ```

- **Vector Norm**:
  ```cpp
  auto vector_norm = norm(a);
  auto squared_norm = squared_norm(a);
  ```


Matrix Operations
-----------------


SQUINT provides essential matrix operations:

- **Trace**:
  ```cpp
  auto matrix_trace = trace(A);
  ```


Statistical Functions
---------------------


SQUINT includes statistical functions for tensors:

- **Mean**:
  ```cpp
  auto tensor_mean = mean(A);
  ```

- **Sum**:
  ```cpp
  auto tensor_sum = sum(A);
  ```

- **Min and Max**:
  ```cpp
  auto min_value = min(A);
  auto max_value = max(A);
  ```


Comparison
----------


SQUINT provides an approximate equality function for comparing tensors:

- **Approximate Equality**:
  ```cpp
  bool are_equal = approx_equal(A, B, tolerance);
  ```


Tensor Contraction (for dynamic tensors)
----------------------------------------


SQUINT supports tensor contraction operations:

- **Tensor Contraction**:
  ```cpp
  auto contracted = contract(A, B, contraction_pairs);
  ```

