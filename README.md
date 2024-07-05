# Squint C++ Library

## Overview

Squint is a C++ library that provides compile-time dimensional analysis and unit conversion capabilities. It allows developers to work with physical quantities in a type-safe manner, preventing common errors related to unit mismatches and improving code readability and maintainability.

## Project Map

### Directory Structure

```
squint/
|-- include/
|   |-- squint/
|       |-- dimension.hpp
|       |-- quantity.hpp
```

### File Contents

include/squint/dimension.hpp

- Defines the squint namespace
- Implements compile-time dimensional types
- Contains:
  - `rational` concept
  - `dimensional` concept
  - `dimension` struct
  - `dim_mult`, `dim_div`, `dim_pow`, `dim_root` structs
  - `dimensions` namespace with common dimension definitions

include/squint/quantity.hpp

- Builds on dimension.hpp
- Implements compile-time quantity types
- Contains:
  - `quantity` class template
  - `units` namespace with various unit types
  - `constants` namespace with mathematical and physical constants

CMakeLists.txt

- Defines the Squint library as a header-only library
- Sets up installation rules

## Installation

### Manual Installation

Squint is a header-only library. To use it in your project, follow these steps:

1. Copy the include/squint directory to your project's include path.
2. Include the necessary headers in your C++ files:

```cpp
#include <squint/quantity.hpp>
```

### Incorporating into a CMake Project

To use Squint in your CMake project using FetchContent, follow these steps:

1. In your project's `CMakeLists.txt`, add the following near the top of the file:

```cmake
include(FetchContent)

FetchContent_Declare(
    squint
    GIT_REPOSITORY https://github.com/barne856/squint.git
    GIT_TAG main  # or a specific tag/commit
)

FetchContent_MakeAvailable(squint)
```

2. After the FetchContent commands, you can link your targets with squint:

```cmake
add_executable(your_target main.cpp)
target_link_libraries(your_target PRIVATE squint::squint)
```

3. In your C++ files, include the Squint headers:

```cpp
#include <squint/quantity.hpp>
```

## Usage

### Example 1: Basic Quantity Operations

```cpp
#include <squint/quantity.hpp>
#include <iostream>

int main() {
    using namespace squint::units;
    
    auto distance = length::meters(100.0);
    auto time = time::seconds(10.0);
    
    auto velocity = distance / time;
    
    std::cout << "Velocity: " << velocity.value() << " m/s" << std::endl;
    
    auto speed_kph = velocity.as<kilometers_per_hour_t>();
    std::cout << "Speed: " << speed_kph << " km/h" << std::endl;
    
    return 0;
}
```

### Example 2: Using Physical Constants

```cpp
#include <squint/quantity.hpp>
#include <iostream>

int main() {
    using namespace squint::constants;
    
    auto c = si_constants<double>::c;
    std::cout << "Speed of light: " << c.value() << " m/s" << std::endl;
    
    auto G = si_constants<double>::G;
    std::cout << "Gravitational constant: " << G.value() << " m^3 kg^-1 s^-2" << std::endl;
    
    return 0;
}
```

## Features

### Dimensional Analysis

- Compile-time checking of dimensional consistency
- Support for all seven SI base dimensions (length, time, mass, temperature, electric current, amount of substance, luminous intensity)
- Operations on dimensions: multiplication, division, power, and root
- Predefined dimensions for common physical quantities (velocity, acceleration, force, energy, etc.)

### Quantity System

- Type-safe representation of physical quantities
- Arithmetic operations: addition, subtraction, multiplication, division
- Comparison operations
- Power and root operations
- Unit conversions
- Support for different underlying value types (int, float, double)

### Error Checking

- Optional compile-time and runtime error checking
- Detects and prevents common errors such as:
  - Integer overflow
  - Division by zero
  - Floating-point underflow

### Constants

- Mathematical constants (pi, e, sqrt(2), etc.)
- Physical constants (speed of light, Planck constant, gravitational constant, etc.)

### Unit Conversions

- Built-in conversions for common units:
  - Length: meters, feet, inches, kilometers, miles
  - Time: seconds, minutes, hours, days
  - Temperature: Kelvin, Celsius, Fahrenheit
  - Mass: kilograms, grams, pounds
  - Velocity: meters per second, kilometers per hour, miles per hour
  - Area: square meters, square feet, acres
  - Volume: cubic meters, liters, gallons
  - Force: newtons, pounds-force
  - Pressure: pascals, bars, psi
  - Energy: joules, kilowatt-hours
  - Power: watts, horsepower

## Advanced Usage

### Working with Mixed types

Squint supports operations between quantities with different underlying types (e.g., int, float, double). The result type is determined by the usual C++ type promotion rules.

```cpp
quantity<double, length> l_double(5.0);
quantity<float, length> l_float(3.0F);
quantity<int, length> l_int(2);

auto result = l_double + l_float + l_int; // result is quantity<double, length>
```

### Error Checking

Squint provides two levels of error checking:

1. `error_checking_disabled` (default): No runtime checks are performed.
2. `error_checking_enabled`: Runtime checks for overflow, division by zero, and underflow are performed.

You can specify the error checking mode when declaring a quantity:

```cpp
quantity<int, length, error_checking_enabled> safe_length(5);
quantity<int, length, error_checking_disabled> fast_length(5);
```

### Working with Constants

Squint provides a set of mathematical and physical constants that can be used in calculations:

```cpp
auto circle_area = constants::math_constants<double>::pi * length_t<double>::meters(2.0).pow<2>();
auto energy = mass_t<double>::kilograms(1.0) * constants::si_constants<double>::c.pow<2>();
```

## API Reference

`squint::quantity<T, D, ErrorChecking>`

The `quantity` class template is the core of the squint library, representing a physical quantity with a value and a dimension.

Template Parameters

- `T`: The underlying arithmetic type (e.g., `float`, `double`)
- `D`: The dimension type
- `ErrorChecking`: Error checking policy (`error_checking_enabled` or `error_checking_disabled`)

Methods

- `constexpr T value() const noexcept`: Returns the underlying value of the quantity
- `template <template <typename, typename> typename TargetUnit, typename TargetErrorChecking = ErrorChecking> constexpr auto as() const`: Converts the quantity to a different unit
- `template <int N> constexpr auto pow() const`: Raises the quantity to the power of N
- `template <int N> auto root() const`: Takes the Nth root of the quantity
- `constexpr auto sqrt() const`: Takes the square root of the quantity

Operators

- Arithmetic operators (`+`, `-`, `*`, `/`)
- Comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`)

`squint::units` Namespace

The `units` namespace contains various unit types and conversion functions. Some of the key unit types include:

- `length_t<T, ErrorChecking>`
- `time_t<T, ErrorChecking>`
- `mass_t<T, ErrorChecking>`
- `temperature_t<T, ErrorChecking>`
- `velocity_t<T, ErrorChecking>`
- `acceleration_t<T, ErrorChecking>`
- `force_t<T, ErrorChecking>`
- `energy_t<T, ErrorChecking>`
- `power_t<T, ErrorChecking>`

Each unit type provides static methods for creating quantities in different units. For example:

```cpp
auto len = length_t<double>::meters(5.0);
auto spd = velocity_t<double>::kilometers_per_hour(60.0);
```

`squint::constants` Namespace

The `constants` namespace provides template structs with various mathematical and physical constants:

- `math_constants<T>`
- `si_constants<T>`
- `astro_constants<T>`
- `atomic_constants<T>`

These can be used to access constants like π, speed of light, gravitational constant, etc.