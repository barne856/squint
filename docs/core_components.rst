
Core Components
===============


Dimension System
----------------


The dimension system in SQUINT uses `std::ratio` for compile-time fraction representation of physical dimensions. Each dimension is represented by seven `std::ratio` values, corresponding to the seven SI base units:

1. Length (L)
2. Time (T)
3. Mass (M)
4. Temperature (K)
5. Electric Current (I)
6. Amount of Substance (N)
7. Luminous Intensity (J)

New dimensions can be created by combining these base dimensions. For example, the dimension of force (F) can be represented as:

:math:`F = M \cdot L \cdot T^{-2}`

.. code-block:: cpp

   using L = dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
   using T = dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
   using M = dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
   
   using velocity_dim = dim_div_t<L, T>;
   using acceleration_dim = dim_div_t<velocity_dim, T>;
   using force_dim = dim_mult_t<M, acceleration_dim>;


The use of std::ratio allows for fractional exponents in the dimension representation.

Quantity System
---------------


The `quantity` class template is the core of SQUINT's quantity system, representing values with associated dimensions:

.. code-block:: cpp

   template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
   class quantity;

Where:

- `T` is the underlying arithmetic type (e.g., `float`, `double`, `int`)
- `D` is the dimension type
- `ErrorChecking` is the error checking policy

SQUINT provides alias templates for common quantity types to enhance readability:

.. code-block:: cpp

   template <typename T> using length_t = quantity<T, dimensions::L>;
   template <typename T> using time_t = quantity<T, dimensions::T>;
   template <typename T> using velocity_t = quantity<T, dimensions::velocity_dim>;
   // ... more aliases for other quantities

Quantities in SQUINT have all the properties of built-in arithmetic types, allowing for intuitive usage in calculations. For instance, calculating acceleration:

.. code-block:: cpp

   // Quantities can be used in mathematical functions
   auto acceleration = units::meters(9.81) / (units::seconds(1) * units::seconds(1));

Importantly, dimensionless quantities can be used interchangeably with built-in arithmetic types, providing a seamless integration with existing code:

.. code-block:: cpp

   auto dimensionless_value = distance / distance;  // Results in a dimensionless quantity
   double scalar_value = 2.0 * dimensionless_value;  // No explicit conversion needed

All mathematical operations on quantities are performed in terms of the base unit. Conversion to/from specific units is handled during construction from non-quantity types or through explicit calls to the `unit_value` method.

It is important to note that the size of any quantity type is exactally the size of it's underlying arithmetic type. This can be important for some applications. For example:

.. code-block:: cpp

   sizeof(float) == sizeof(length_t<float>);

Tensor System
-------------


SQUINT's tensor system is built around a single, flexible `tensor` class with a policy-based design, supporting both fixed and dynamic shapes:

.. code-block:: cpp

   template <typename T, typename Shape, typename Strides = strides::column_major<Shape>,
             error_checking ErrorChecking = error_checking::disabled,
             ownership_type OwnershipType = ownership_type::owner,
             memory_space MemorySpace = memory_space::host>
   class tensor;

Key features of the tensor system include:

- Single class design for both fixed and dynamic shapes
- Compile-time optimizations for fixed shapes
- Runtime flexibility for dynamic shapes
- Configurable error checking
- Flexible memory ownership (owner or reference)
- Support for different memory spaces
- Column-major default layout for construction and iteration

The library includes aliases for common tensor types to improve code readability:

.. code-block:: cpp

   template <typename T> using vec3_t = tensor<T, shape<3>>;
   template <typename T> using mat3_t = tensor<T, shape<3, 3>>;
   // ... more aliases for other tensor types
   
   using vec3 = vec3_t<float>;
   using mat3 = mat3_t<float>;
   // ... more type-specific aliases

It is important to note that the size of any fixed shape tensor type is exactally the size of it's elements. This can be important for some applications. For example:

.. code-block:: cpp

   sizeof(vec3) == 3 * sizeof(float);
