
Advanced Features
=================



Error Checking
--------------


SQUINT provides optional runtime error checking for both quantities and tensors. This feature can be enabled or disabled at compile-time:

.. code-block::

   using checked_quantity = quantity<double, length, error_checking::enabled>;
   tensor<float, shape<3, 3>, strides::column_major<shape<3, 3>>, error_checking::enabled> checked_tensor;

When error checking is enabled, SQUINT performs various runtime checks:

- For quantities:

  - Overflow and underflow checks in arithmetic operations
  - Division by zero checks
  - Dimension compatibility checks in operations

- For tensors:

  - Bounds checking for element access
  - Shape compatibility checks in operations
  - Dimension checks for linear algebra operations

Enabling error checking can be valuable during development and debugging, while it can be disabled for maximum performance in production builds.


Unit Conversions
----------------


SQUINT supports seamless unit conversions within the same dimension:

.. code-block::

   auto meters = length_t<double>::meters(5.0);
   auto feet = convert_to<units::feet_t>(meters);
   auto inches = convert_to<units::inches_t>(meters);
   
   // Unit conversions can also be done directly in calculations
   auto speed = length_t<double>::kilometers(60.0) / time_t<double>::hours(1.0);
   auto speed_mph = convert_to<units::miles_per_hour_t>(speed);


Constants
---------


SQUINT includes a comprehensive set of physical and mathematical constants for example:

.. code-block::

   // Physical constants
   auto c = si_constants<double>::c;  // Speed of light
   auto G = si_constants<double>::G;  // Gravitational constant
   auto h = si_constants<double>::h;  // Planck constant
   
   // Mathematical constants
   auto pi = math_constants<double>::pi;  // Pi
   auto e = math_constants<double>::e;   // Euler's number
   
   // Astronomical constants
   auto AU = astro_constants<double>::AU;  // Astronomical Unit
   auto solar_mass = astro_constants<double>::solar_mass;  // Solar mass
   
   // Atomic constants
   auto electron_mass = atomic_constants<double>::electron_mass;  // Electron mass

These constants and more are implemented as `constant_quantity_t` types, ensuring proper dimensional analysis in calculations.


Tensor Views with Step Sizes
----------------------------


SQUINT supports creating tensor views with custom step sizes for both fixed and dynamic shape tensors, allowing for more flexible and efficient data access patterns. This feature is particularly useful for operations like strided slicing or accessing every nth element along a dimension.


API for Fixed Shape Tensors
---------------------------


For fixed shape tensors, SQUINT provides a template method that determines the step sizes at compile-time:

.. code-block::

   template <typename SubviewShape, typename StepSizes>
   auto subview(const index_type& start_indices);

Here, `SubviewShape` defines the shape of the resulting view, and `StepSizes` defines the step sizes along each dimension.

Usage example for fixed shape tensors:

.. code-block::

   // Create a 4x4 matrix
   mat4 A = mat4::random(0.0, 1.0);
   
   // Create a view that takes every 2nd element in both dimensions
   auto strided_view = A.subview<shape<2, 2>, seq<2, 2>>({0, 0});
   
   // strided_view now represents:
   // [A(0,0) A(0,2)]
   // [A(2,0) A(2,2)]
   
   // Create a view of the main diagonal
   auto diagonal_view = A.subview<shape<4>, seq<5>>({0, 0});
   
   // diagonal_view now represents the main diagonal of A:
   // [A(0,0) A(1,1) A(2,2) A(3,3)]


API for Dynamic Shape Tensors
-----------------------------


For dynamic shape tensors, SQUINT provides a method that takes runtime arguments for the subview shape, start indices, and step sizes:

.. code-block::

   auto subview(const index_type& subview_shape, const index_type& start_indices, const index_type& step_sizes);

Usage example for dynamic shape tensors:

.. code-block::

   // Create a 10x10x10 tensor
   dynamic_tensor<float> B({10, 10, 10});
   
   // Create a 3x3x3 view with elements spaced 3 apart in each dimension
   auto custom_view = B.subview({3, 3, 3}, {1, 1, 1}, {3, 3, 3});

When using views with step sizes, keep in mind:

- The resulting view is not guaranteed to be contiguous in memory.
- Operations on these views may be less efficient than on contiguous data, depending on the hardware and BLAS backend.
- For fixed shape tensors, the shape and step sizes are checked at compile-time, providing additional type safety.
- For dynamic shape tensors, the shape of the resulting view is determined by the `subview_shape` parameter, not by the original tensor's shape and the step sizes.

