Quantities and Units
====================

SQUINT provides a robust system for handling physical units and dimensions, ensuring type safety and dimensional correctness in your calculations. This page explains how to use units in SQUINT and provides examples of common operations.

Basic Usage
-----------

In SQUINT, quantities are represented using the `quantity` class template. Each quantity has a value and an associated dimension. Quantity types like `length` and `time` are specializations of the `quantity` class with specific dimensions and units types like units::meters are derived classes of those quantity types. Here's a basic example:

.. code-block:: cpp

    #include <squint/quantity.hpp>
    
    using namespace squint::units;

    // Create a length of 5 meters
    auto length = meters(5.0);

    // Create a time of 2 seconds
    auto time = seconds(2.0);

    // Calculate velocity
    auto velocity = length / time;

    std::cout << "Velocity: " << velocity << " m/s" << std::endl;

.. note::
    When a quantity is constructed without specifying a unit (e.g., `length(5.0)` instead of `units::meters(5.0)`), it is constructed using the base unit type for that quantity.

Predefined Units
----------------

SQUINT provides a wide range of predefined units for common physical quantities. Here are some examples:

- Length: meters, kilometers, feet, inches, etc.
- Time: seconds, minutes, hours, days, etc.
- Mass: kilograms, grams, pounds, etc.
- Velocity: meters_per_second, kilometers_per_hour, etc.
- Acceleration: meters_per_second_squared, etc.
- Force: newtons, pound_force, etc.
- Energy: joules, kilowatt_hours, etc.

You can create quantities using these predefined units:

.. code-block:: cpp

    auto distance = units::kilometers(10.0);
    auto mass = units::kilograms(75.0);
    auto speed = units::miles_per_hour(60.0);

Unit Literals
-------------

SQUINT provides user-defined literals for many common units, allowing for a more concise and readable way to create quantities. To use unit literals, you need to include the appropriate header and use the `squint::literals` namespace:

.. code-block:: cpp

    #include <squint/quantity/unit_literals.hpp>

    using namespace squint::literals;


Here are some examples of how to use unit literals:

.. code-block:: cpp

    auto length = 5.0_m;      // 5 meters
    auto time = 2.5_s;        // 2.5 seconds
    auto mass = 75.0_kg;      // 75 kilograms
    auto temperature = 20.0_C; // 20 degrees Celsius
    auto angle = 45.0_deg;    // 45 degrees
    auto force = 10.0_N;      // 10 newtons
    auto pressure = 1.0_atm;  // 1 atmosphere

Using unit literals can make your code more readable and less prone to errors, as the units are clearly specified right next to the numeric values.

Here's an example that combines unit literals with calculations:

.. code-block:: cpp

    auto distance = 100.0_km;
    auto time = 2.0_h;
    auto speed = distance / time;

    std::cout << "Average speed: " << speed << std::endl;  // Output will be in base units of m/s

    auto acceleration = 9.81_mps / 1.0_s;
    std::cout << "Acceleration: " << acceleration << std::endl;  // Output will be in m/s^2

Unit literals are particularly useful when working with mixed units:

.. code-block:: cpp

    auto total_length = 5.0_m + 30.0_cm + 2.0_in;
    std::cout << "Total length: " << total_length << std::endl;  // Output will be in meters

SQUINT automatically handles the unit conversions, ensuring that the calculations are correct regardless of the input units.

Unit Conversions
----------------

SQUINT allows easy conversion between compatible units:

.. code-block:: cpp

    auto length_m = units::meters(1000.0);
    auto length_km = convert_to<units::kilometers_t>(length_m);
    units::kilometers length_km_2 = length_m; // Equivalent to the above

    std::cout << length_m.unit_value() << " meters is " << length_km.unit_value() << " kilometers" << std::endl;

Arithmetic with Units
---------------------

You can perform arithmetic operations with quantities, and SQUINT will handle the dimensional analysis:

.. code-block:: cpp

    auto distance = unit::meters(100.0);
    auto time = unit::seconds(10.0);

    auto velocity = distance / time;
    auto acceleration = velocity / time;

    std::cout << "Acceleration: " << acceleration << " m/s^2" << std::endl;

Creating Custom Units
---------------------

If you need a unit that's not predefined, you can create custom units:

.. code-block:: cpp

    // Define a new unit for area (square meters)
    template <typename T> using square_meters_t = unit<T, dim_pow_t<dimensions::L, 2>>;

    // Define a new unit for area (square feet)
    template <typename T>
    using square_feet_t = unit<T, dim_pow_t<dimensions::L, 2>, static_cast<T>(FEET_TO_METERS *FEET_TO_METERS)>;

    // Create a quantity using the new unit
    auto area = square_feet_t<double>(50.0);

Using Units with Tensors
------------------------

SQUINT allows you to use units with tensors, enabling type-safe calculations in linear algebra and physics simulations:

.. code-block:: cpp

    #include <squint/tensor.hpp>

    // Create a 3D vector representing position
    vec3_t<length> position{
        units::meters(1.0),
        units::meters(2.0),
        units::meters(3.0)
    };

    // Create a 3D vector representing velocity
    vec3_t<velocity> vel{
        units::meters_per_second(4.0),
        units::meters_per_second(5.0),
        units::meters_per_second(6.0)
    };

    // Calculate displacement after 2 seconds
    auto t = units::seconds(2.0);
    auto displacement = vel * t;

    std::cout << "Displacement: " << displacement << std::endl;

Error Checking
--------------

SQUINT provides both compile-time and runtime error checking for quantities. The compile-time checks ensure dimensional correctness, while runtime checks (when enabled) catch arithmetic errors.

Compile-time Checks
^^^^^^^^^^^^^^^^^^^

Compile-time checks prevent operations between incompatible dimensions. For example:

.. code-block:: cpp

    length l = units::meters(5.0);
    time t = units::seconds(2.0);
    
    auto sum = l + t;  // This will not compile: cannot add length and time

Runtime Checks
^^^^^^^^^^^^^^

SQUINT also provides optional runtime error checking. This can be enabled by using the `error_checking::enabled` policy:

.. code-block:: cpp

    using checked_length = quantity<int, dimensions::L, error_checking::enabled>;

When runtime error checking is enabled, SQUINT will throw exceptions for various arithmetic errors:

1. Overflow:

.. code-block:: cpp

    checked_length l1(std::numeric_limits<int>::max());
    checked_length l2(1);
    auto sum = l1 + l2;  // This will throw std::overflow_error

2. Underflow:

.. code-block:: cpp

    checked_length l1(std::numeric_limits<int>::min());
    checked_length l2(1);
    auto diff = l1 - l2;  // This will throw std::underflow_error

3. Division by zero:

.. code-block:: cpp

    checked_length l(10);
    auto result = l / 0;  // This will throw std::domain_error

4. Multiplication overflow:

.. code-block:: cpp

    checked_length l(1000000);
    auto product = l * 1000000;  // This will throw std::overflow_error


Basic Operations
----------------


SQUINT provides a comprehensive set of mathematical operations for quantities:

- **Absolute Value**:

.. code-block:: cpp

   auto abs_value = abs(quantity);

- **Square Root**:

.. code-block:: cpp

   auto sqrt_value = sqrt(quantity);

- **Nth Root**:
  
.. code-block:: cpp

   auto nth_root = root<N>(quantity);

- **Exponential** (for dimensionless quantities):
  
.. code-block:: cpp

   auto exp_value = exp(dimensionless_quantity);

- **Logarithm** (for dimensionless quantities):
  
.. code-block:: cpp

   auto log_value = log(dimensionless_quantity);

- **Power**:
  
.. code-block:: cpp

   auto powered_value = pow<N>(quantity);


Trigonometric Functions
-----------------------


For dimensionless quantities, SQUINT provides standard trigonometric functions for dimensionless quantities:

- **Sine, Cosine, Tangent**:

.. code-block:: cpp

   auto sin_value = sin(angle);
   auto cos_value = cos(angle);
   auto tan_value = tan(angle);

- **Inverse Trigonometric Functions**:
  
.. code-block:: cpp

   auto asin_value = asin(dimensionless_quantity);
   auto acos_value = acos(dimensionless_quantity);
   auto atan_value = atan(dimensionless_quantity);

- **Two-argument Arctangent**:
  
.. code-block:: cpp

   auto atan2_value = atan2(y, x);


Hyperbolic Functions
--------------------


SQUINT also includes hyperbolic functions for dimensionless quantities:

- **Hyperbolic Sine, Cosine, Tangent**:

.. code-block:: cpp

   auto sinh_value = sinh(dimensionless_quantity);
   auto cosh_value = cosh(dimensionless_quantity);
   auto tanh_value = tanh(dimensionless_quantity);

- **Inverse Hyperbolic Functions**:
  
.. code-block:: cpp

   auto asinh_value = asinh(dimensionless_quantity);
   auto acosh_value = acosh(dimensionless_quantity);
   auto atanh_value = atanh(dimensionless_quantity);


Comparison
----------


In addition to the standard comparision operators, SQUINT provides an approximate equality function for comparing quantities:

- **Approximate Equality**:

.. code-block:: cpp

   bool are_equal = approx_equal(quantity1, quantity2, epsilon);