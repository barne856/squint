# Squint C++ Library

## Table of Contents

1. [Introduction](#introduction)
2. [Design Principles](#design-principles)
3. [Project Map](#project-map)
   - [Directory Structure](#directory-structure)
   - [File Contents](#file-contents)
4. [Walkthrough of Core Types](#walkthrough-of-core-types)
   - [Quantity Types](#quantity-types)
   - [Tensor Types](#tensor-types)
   - [Combining Quantities and Tensors](#combining-quantities-and-tensors)
5. [Installation](#installation)
   - [Manual Installation](#manual-installation)
   - [Incorporating into a CMake Project](#incorporating-into-a-cmake-project)
6. [Usage](#usage)
   - [Example 1: Basic Quantity Operations](#example-1-basic-quantity-operations)
   - [Example 2: Using Physical Constants](#example-2-using-physical-constants)
   - [Example 3: Basic Tensor Operations](#example-3-basic-tensor-operations)
   - [Example 4: Tensor Static Methods](#example-4-tensor-static-methods)
7. [Features](#features)
   - [Dimensional Analysis](#dimensional-analysis)
   - [Quantity System](#quantity-system)
   - [Error Checking](#error-checking)
   - [Constants](#constants)
   - [Unit Conversions](#unit-conversions)
   - [Tensor Operations](#tensor-operations)
8. [Advanced Usage](#advanced-usage)
   - [Working with Mixed types](#working-with-mixed-types)
   - [Error Checking](#error-checking-1)
   - [Working with Constants](#working-with-constants)
   - [Working with Tensors](#working-with-tensors)
9. [API Reference](#api-reference)


## Introduction

SQUINT (Static Quantities in Tensors) is a modern, header-only C++ library designed to bring robust dimensional analysis, unit conversion, and linear algebra operations to scientific and engineering applications. By leveraging C++'s template metaprogramming capabilities, Squint provides a powerful set of tools that enhance code safety, readability, and expressiveness without compromising performance.

Key features of Squint include:

1. **Dimensional Analysis**: Squint implements a compile-time dimensional analysis system that catches unit-related errors at compile-time, preventing common mistakes in scientific and engineering calculations.
2. **Unit Conversions**: The library offers a wide range of predefined units and easy-to-use conversion functions, simplifying the often error-prone process of unit conversion.
3. **Quantity System**: Squint's quantity system allows for intuitive representation of physical quantities, complete with unit information, enabling natural arithmetic operations between quantities of compatible dimensions. Qunatities work seamlessly with built in arithmetic types.
4. **Tensor Operations**: The library provides both fixed-size and dynamic-size tensor classes (N-D arrays), supporting various linear algebra operations, element-wise arithmetic, and advanced indexing, views, and slicing capabilities. Tensors are designed to work seamlessly with both built in types and `quantity` types.
5. **Error Checking**: Optional runtime error checking for operations like overflow, underflow, division by zero, and bounds checking adds an extra layer of safety to numerical computations.
6. **Constants and Common Functions**: Squint includes a comprehensive set of mathematical and physical constants, as well as functions commonly used in scientific computing. Common math functions such as those in `<cmath>` are defined to work seamlessly with `quantity` types.
7. **Performance**: Despite its high-level abstractions, Squint is designed with performance in mind, utilizing template metaprogramming techniques to minimize runtime overhead.

Squint is particularly useful for:

- Scientific simulations and modeling
- Engineering calculations and analysis
- Data processing and visualization in technical fields
- Educational purposes, demonstrating concepts of dimensional analysis and tensor operations

By using Squint, developers can write more expressive, safer code that closely mirrors the mathematical and physical concepts they're working with, while catching potential errors early in the development process.

## Design Principles

Squint is built on several core principles that guide its design and implementation:

1. **Type Safety**: Squint leverages C++'s strong type system to catch errors at compile-time rather than runtime. This principle is at the heart of its dimensional analysis system and tensor operations.

```cpp
length l = length::meters(5.0);
time t = time::seconds(2.0);
auto v = l / t;  // type 'velocity' inferred
// auto e = l + t;  // Compile-time error: can't add length and time
```

2. **Expressiveness**: The library aims to allow users to write code that closely mirrors mathematical and physical equations.

```cpp
// Calculate kinetic energy: E = 1/2 * m * v^2
mass m = mass::kilograms(2.0);
velocity v = velocity::meters_per_second(3.0);
energy E = 0.5 * m * v.pow<2>();
```

3. **Flexibility**: Squint is designed to be flexible, allowing users to work with different numeric types (int, float, double) or quantities and providing both fixed-size and dynamic-size tensors to suit various use cases.

```cpp
// Using different numeric types
auto l_float = length_t<float>(5);
auto l_double = length_t<double>(5);
auto l_int = length_t<int>(5);
// Fixed-size tensor of integer lengths
auto fixed_matrix = mat3<length_t<int>>();
// Dynamic-size tensor of doubles
auto dynamic_matrix = dtens<double>();
```

4. **Performance**: Despite its high-level abstractions, Squint is implemented using template metaprogramming techniques to minimize or eliminate runtime overhead, aiming to be as performant as hand-written, dimension-specific code.

```cpp
// built in types
float d = 5.0;
float t = 2.0;
float v = d / t;
float a = v / t;
// squint types are zero overhead abstractions and produce
// equivalent performance to using built in types
length d = length::meters(5.0);
time t = time::seconds(2.0);
velocity v = d / t;
acceleration a = v / t;
```
5. **Extensibility**: The library is designed to be extensible, allowing users to define their own units, dimensions, and potentially even custom tensor types.

```cpp
// Define a new dimension
using information = dimensions::dimensionless;

// Define a new unit
template <typename T, squint::error_checking ErrorChecking = squint::error_checking::disabled>
struct information_t : squint::unit_t<T, information, ErrorChecking> {
    using squint::unit_t<T, information, ErrorChecking>::unit_t;
    static constexpr information_t<T, ErrorChecking> bits(T value) { return information_t<T, ErrorChecking>(value); }
    static constexpr information_t<T, ErrorChecking> bytes(T value) { return information_t<T, ErrorChecking>(value * 8); }
};
```

## Project Map

### Directory Structure

```
squint/
|-- include/
|   |-- squint/
|       |-- dimension.hpp
|       |-- quantity.hpp
|       |-- tensor.hpp
|       |-- tensor_base.hpp
|       |-- fixed_tensor.hpp
|       |-- dynamic_tensor.hpp
|       |-- tensor_view.hpp
|       |-- iterable_tensor.hpp
|       |-- linear_algebra.hpp
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
  - `math` namespace with mathematical functions for use with quantity types

include/squint/tensor.hpp and related files

- Implement tensor operations and abstractions
- Contains:
  - `fixed_tensor` and `dynamic_tensor` class templates
  - Tensor view classes for efficient sub-tensor operations
  - Iterators for element-wise access
  - Basic linear algebra operations

## Walkthrough of Core Types

### Quantity Types

At the heart of Squint's dimensional analysis system is the `quantity` class template. Here's how it works:

```cpp
template <typename T, dimensional D, error_checking ErrorChecking = error_checking::disabled>
class quantity;
```

- `T`: The underlying numeric type (e.g., `float`, `double`, `int`)
- `D`: The dimension type (e.g., `length`, `time`, `mass`)
- `ErrorChecking`: An enum to enable or disable runtime error checking

#### Using Quantity Types

There are many predefined convenience types such as `length`, `velocity`, etc. For a full list see the `quantity.hpp` file. These convenience types have error checking disabled and have underlying type of `float`, but you can specify another type by using the types that end with `_t`. For example `length_t<double>`.

1. **Creating Quantities**:

   ```cpp
   using namespace squint::units;
   
   auto l = length(5.0); // or length::meters(5), length::feet(5), etc. to specify units
   auto t = time(2.0);
   ```

2. **Arithmetic with Quantities**:

   ```cpp
   auto velocity = l / t;
   auto acceleration = v / t;
   ```

3. **Unit Conversions**:

   ```cpp
   float length_in_feet = l.as<feet_t>();
   float speed_in_kph = v.as<kilometers_per_hour_t>();
   ```

4. **Error Checking**:

   ```cpp
   quantity<int, dimensions::length, error_checking::enabled> safe_length(10000000);
   safe_length *= 1000; // This will throw an overflow error at runtime
   ```

If an error checked type is combined in an expression with a non error checked type, the result will be an error checked type. This allows error checking to propogate through a program by just inserting a single error checked type.

### Tensor Types

Squint provides both fixed-size and dynamic-size tensor types. The main tensor types are:

1. `fixed_tensor`: For tensors whose dimensions are known at compile-time
2. `dynamic_tensor`: For tensors whose dimensions are determined at runtime

#### Using Tensor Types

There are convieince types defined for common tensor shapes, for a complete list see the `fixed_tensor.hpp` and `dynamic_tensor.hpp` files.

1. **Creating Tensors**:

   ```cpp
   // Fixed-size 3x3 matrix
   mat3<float> matrix = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
   
   // Dynamic-size tensor
   dynamic_tensor<double> dynamic_tensor({2, 3, 4}); // 2x3x4 tensor
   ```

2. **Accessing Elements**:

   ```cpp
   float element = matrix[1, 2];
   double dynamic_element = dynamic_tensor.at({1, 2, 3});
   ```

3. **Tensor Operations**:

   ```cpp
   auto sum = matrix + matrix;
   auto scaled = matrix * 2.0f;
   ```

4. **Common Tensors**:

   ```cpp
   auto identity = mat4<float>::I();
   auto random_matrix = mat3<double>::random(0.0, 1.0);
   ```

5. **Tensor Views and Slicing**:

   ```cpp
   auto view = matrix.view();
   auto subview = matrix.subview<2, 2>(slice{0, 2}, slice{1, 2});
   ```

### Combining Quantities and Tensors

One of the powerful features of Squint is the ability to create tensors of quantities, allowing for type-safe multi-dimensional physical calculations.

```cpp
// A 3D vector of times
vec3<time> times = {
    time::seconds(1.0),
    time::seconds(2.0),
    time::seconds(3.0)
};

// A 3x3 matrix of velocities
mat3<velocity> velocity_field = {
    velocity::meters_per_second(1.0), velocity::meters_per_second(2.0), velocity::meters_per_second(3.0),
    velocity::meters_per_second(4.0), velocity::meters_per_second(5.0), velocity::meters_per_second(6.0),
    velocity::meters_per_second(7.0), velocity::meters_per_second(8.0), velocity::meters_per_second(9.0)
};

// Perform operations
auto lengths = velocity_field * times; // Results in a vec3<quantity<double, length>>
```

In this example, we've created a 3D time vector and a 3x3 matrix of velocities. When we multiply the velocity field by the time vector, we get a new position vector since dimensionally (L / T) * T = L. The library ensures that the dimensions are correct and handles the unit calculations automatically.

This combination of quantities and tensors allows for expressive and type-safe representations of complex physical systems, such as stress tensors in material science, electromagnetic field tensors in physics, or multi-dimensional data in any scientific or engineering discipline.

Tensors can also have error checking enabled. This is separate from qunatity error checking and essentially only enables bounds checking at runtime. It is possbile to have error checked tensors with non error checked quantities or error checked quantities and in expressions the usual type deduction rules for quantities apply.

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
#include <squint/tensor.hpp>
```

### Building Tests


## On Windows

Install MSVC, CMake >= 3.28, Clang >= 15, MKL (optional), Intel fortran compiler (optional if MKL is used as backend)

You must set all env vars with:

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
```

Then:

```
mkdir build
cd build
```

For OpenBLAS backend:

```
cmake -DCMAKE_Fortran_COMPILER=ifx -DBLAS_BACKEND=OpenBLAS -T ClangCL ..
```

For MKL backend

```
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DBLAS_BACKEND=MKL ..
```

```
cd ..
cmake --build ./build
```

## On Linux

Install CMake >= 3.28, Clang >= 15, MKL (optional), gfortran (optional if MKL is used as backend)

Configure env vars like the devcontainer

```
mkdir build
cd build
```

For OpenBLAS backend:

```
cmake -DBLAS_BACKEND=OpenBLAS ..
```

For MKL backend

```
cmake -DBLAS_BACKEND=MKL ..
```

```
cd ..
cmake --build ./build
```

## Run the tests:

```
cd build
ctests
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

### Example 3: Basic Tensor Operations

```cpp
#include <squint/tensor.hpp>
#include <iostream>

int main() {
    using namespace squint;

    // Create a 2x3 fixed tensor
    mat2x3<int> t{{1, 2, 3, 4, 5, 6}};

    std::cout << "Tensor: " << t << std::endl;

    // Access elements
    std::cout << "Element at (1, 2): " << t[1, 2] << std::endl;

    // Create a view
    auto view = t.view();

    // Create a subview
    auto subview = t.subview<2, 2>(slice{0, 2}, slice{1, 2});

    return 0;
}
```

### Example 4: Tensor Static Methods

```cpp
#include <squint/tensor.hpp>
#include <iostream>

int main() {
    using namespace squint;

    // Create an identity matrix
    auto identity = mat3<float>::I();
    std::cout << "3x3 Identity Matrix:\n" << identity << std::endl;

    // Create a diagonal matrix
    auto diagonal = mat4<double>::diag(5.0);
    std::cout << "4x4 Diagonal Matrix with 5 on the diagonal:\n" << diagonal << std::endl;

    // Create a tensor filled with ones
    auto ones = mat2x3<int>::ones();
    std::cout << "2x3 Tensor filled with ones:\n" << ones << std::endl;

    // Create a random tensor
    auto random = mat2<float>::random(0.0f, 1.0f);
    std::cout << "2x2 Random Matrix (values between 0 and 1):\n" << random << std::endl;

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

### Tensor Operations

- Fixed-size and dynamic-size tensors
- Row-major and column-major memory layouts
- Multi-dimensional indexing and slicing
- Tensor views for efficient sub-tensor operations
- Iterators for element-wise access
- Basic linear algebra operations

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

1. `error_checking::disabled` (default): No runtime checks are performed.
2. `error_checking::enabled`: Runtime checks for overflow, division by zero, and underflow are performed.

You can specify the error checking mode when declaring a quantity:

```cpp
quantity<int, length, error_checking::enabled> safe_length(5);
quantity<int, length, error_checking::disabled> fast_length(5);
```

### Working with Constants

Squint provides a set of mathematical and physical constants that can be used in calculations:

```cpp
auto circle_area = constants::math_constants<double>::pi * length_t<double>::meters(2.0).pow<2>();
auto energy = mass_t<double>::kilograms(1.0) * constants::si_constants<double>::c.pow<2>();
```

### Working with Tensors

Squint supports both fixed-size and dynamic-size tensors. You can perform various operations on tensors, including element-wise arithmetic, slicing, and basic linear algebra operations.

```cpp
#include <squint/tensor.hpp>
#include <iostream>

int main() {
    using namespace squint;

    // Create two 2x3 tensors
    mat2x3<int> t1{{1, 2, 3, 4, 5, 6}};
    mat2x3<int> t2{{6, 5, 4, 3, 2, 1}};

    // Element-wise addition
    auto sum = t1 + t2;
    std::cout << "Sum: " << sum << std::endl;

    // Scalar multiplication
    auto scaled = t1 * 2;
    std::cout << "Scaled: " << scaled << std::endl;

    // Static helper methods for common tensors
    auto identity = mat3<float>::I();
    std::cout << "3x3 Identity Matrix:\n" << identity << std::endl;

    auto diagonal = vec3<double>::diag(2.0);
    std::cout << "3x3 Diagonal Matrix:\n" << diagonal << std::endl;

    auto random = mat2x3<float>::random(-1.0f, 1.0f);
    std::cout << "2x3 Random Matrix:\n" << random << std::endl;

    return 0;
}
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

`squint::fixed_tensor<T, L, ErrorChecking, Dims...>`

The `fixed_tensor` class template represents a fixed-size tensor with compile-time dimensions.

Template Parameters

- `T`: The underlying data type (e.g., `float`, `double`, `int`)
- `L`: Memory layout (`layout::row_major` or `layout::column_major`)
- `ErrorChecking`: Error checking policy (`error_checking::enabled` or `error_checking::disabled`)
- `Dims...`: Compile-time dimensions of the tensor

`squint::dynamic_tensor<T, ErrorChecking>`

The `dynamic_tensor` class template represents a dynamic-size tensor with runtime-determined dimensions.

Template Parameters

- `T`: The underlying data type (e.g., `float`, `double`, `int`)
- `ErrorChecking`: Error checking policy (`error_checking::enabled` or `error_checking::disabled`)

Both tensor types provide methods for accessing elements, creating views and subviews, and performing basic linear algebra operations.