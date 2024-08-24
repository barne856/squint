Quantities and Units
=====

SQUINT provides a robust system for handling physical units and dimensions, ensuring type safety and dimensional correctness in your calculations. This page explains how to use units in SQUINT and provides examples of common operations.

Basic Usage
-----------

In SQUINT, quantities are represented using the `quantity` class template. Each quantity has a value and an associated dimension. Quantity types like `length` and `time` are specializations of the `quantity` class with specific dimensions set. Here's a basic example:

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

Unit Conversions
----------------

SQUINT allows easy conversion between compatible units:

.. code-block:: cpp

    auto length_m = units::meters(1000.0);
    auto length_km = convert_to<units::kilometer_t>(length_m);

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
    auto displacement = vel * time;

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
