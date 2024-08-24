Units in SQUINT
===============

SQUINT provides a robust system for handling physical units and dimensions, ensuring type safety and dimensional correctness in your calculations. This page explains how to use units in SQUINT and provides examples of common operations.

Basic Usage
-----------

In SQUINT, quantities are represented using the `quantity` class template. Each quantity has a value and an associated dimension. Here's a basic example:

```cpp
#include <squint/quantity.hpp>

using namespace squint;

// Create a length of 5 meters
auto length = length::meters(5.0);

// Create a time of 2 seconds
auto time = time::seconds(2.0);

// Calculate velocity
auto velocity = length / time;

std::cout << "Velocity: " << velocity.value() << " m/s" << std::endl;
```

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

```cpp
auto distance = length::kilometers(10.0);
auto mass = mass::kilograms(75.0);
auto speed = velocity::miles_per_hour(60.0);
```

Unit Conversions
----------------

SQUINT allows easy conversion between compatible units:

```cpp
auto length_m = length::meters(1000.0);
auto length_km = convert_to<units::kilometer_t>(length_m);

std::cout << length_m.value() << " meters is " << length_km.value() << " kilometers" << std::endl;
```

Arithmetic with Units
---------------------

You can perform arithmetic operations with quantities, and SQUINT will handle the dimensional analysis:

```cpp
auto distance = length::meters(100.0);
auto time = time::seconds(10.0);

auto velocity = distance / time;
auto acceleration = velocity / time;

std::cout << "Acceleration: " << acceleration.value() << " m/s^2" << std::endl;
```

Creating Custom Units
---------------------

If you need a unit that's not predefined, you can create custom units:

```cpp
// Define a new unit for area (square meters)
using square_meter_t = unit<dimension_power_t<dimensions::L, 2>>;

// Create a quantity using the new unit
auto area = quantity<double, dimension_power_t<dimensions::L, 2>>::make<square_meter_t>(50.0);
```

Using Units with Tensors
------------------------

SQUINT allows you to use units with tensors, enabling type-safe calculations in linear algebra and physics simulations:

```cpp
#include <squint/tensor.hpp>

// Create a 3D vector representing position
vec3_t<length_t<double>> position{
    length::meters(1.0),
    length::meters(2.0),
    length::meters(3.0)
};

// Create a 3D vector representing velocity
vec3_t<velocity_t<double>> velocity{
    velocity::meters_per_second(4.0),
    velocity::meters_per_second(5.0),
    velocity::meters_per_second(6.0)
};

// Calculate displacement after 2 seconds
auto time = time::seconds(2.0);
auto displacement = velocity * time.value();

std::cout << "Displacement: " << displacement << std::endl;
```

Error Checking
--------------

SQUINT provides optional runtime error checking for quantities. You can enable this feature to catch dimension mismatches and other errors:

```cpp
using checked_length = quantity<double, dimensions::L, error_checking::enabled>;

checked_length length1 = checked_length::meters(5.0);
checked_length length2 = checked_length::meters(3.0);

// This will compile and run correctly
auto sum = length1 + length2;

// This will throw a runtime error
auto invalid = length1 + time::seconds(2.0);
```

By using SQUINT's unit system, you can write more robust and error-free code, catching dimensional errors at compile-time or runtime, depending on your needs.