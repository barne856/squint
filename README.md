# SQUINT Tensor Library

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Building and Testing](#building-and-testing)
   - [Compiler Support](#compiler-support)
   - [Build Instructions](#build-instructions)
   - [CMake Configuration](#cmake-configuration)
   - [BLAS Backends](#blas-backends)
5. [Core Components](#core-components)
   - [Dimension System](#dimension-system)
   - [Quantity System](#quantity-system)
   - [Tensor System](#tensor-system)
6. [Usage](#usage)
   - [Quantity Usage](#quantity-usage)
   - [Tensor Construction](#tensor-construction)
   - [Basic Operations](#basic-operations)
   - [Views and Reshaping](#views-and-reshaping)
   - [Linear Algebra Operations](#linear-algebra-operations)
7. [Advanced Features](#advanced-features)
   - [Error Checking](#error-checking)
   - [Unit Conversions](#unit-conversions)
   - [Constants](#constants)
8. [API Reference](#api-reference)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

## Introduction

SQUINT (Static Quantities in Tensors) is a modern, header-only C++ library designed for compile-time dimensional analysis, unit conversion, and efficient linear algebra operations. It provides a powerful API for enhancing code safety, readability, and expressiveness without compromising performance.

SQUINT combines a robust quantity system for handling physical units and dimensions with a flexible tensor system for numerical computations. This integration allows for type-safe calculations involving physical quantities and efficient matrix operations, making it ideal for scientific computing, engineering simulations, and data analysis tasks.

## Key Features

- Compile-time dimensional analysis for catching unit-related errors early
- Efficient tensor operations with both fixed and dynamic shapes
- Support for physical quantities with units, seamlessly integrated with tensors
- Flexible error checking policies for both quantities and tensors
- Integration with BLAS and LAPACK for high-performance linear algebra
- Header-only design for easy integration into existing projects
- Comprehensive set of mathematical operations for both quantities and tensors
- Support for unit conversions and physical constants
- Column-major default layout for tensors, aligning with common scientific computing practices

## Installation

SQUINT is a header-only library, making it easy to integrate into your projects. To use it:

1. Copy the `include/squint` directory to your project's include path.
2. Include the necessary headers in your C++ files:

```cpp
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>
```

For CMake projects, you can use FetchContent for a more streamlined integration:

```cmake
include(FetchContent)

FetchContent_Declare(
    squint
    GIT_REPOSITORY https://github.com/barne856/squint.git
    GIT_TAG main  # or a specific tag/commit
)

FetchContent_MakeAvailable(squint)

target_link_libraries(your_target PRIVATE squint::squint)
```

## Building and Testing

### Compiler Support

SQUINT leverages modern C++ features and requires a C++23 compliant compiler. Currently supported compilers include:

- GCC (g++) version 12 or later
- Clang version 15 or later

Note: MSVC is partially supported but lacks support for multidimensional subscript operators.

### Build Instructions

1. Ensure you have CMake version 3.28 or later and a supported compiler installed.
2. Optionally install MKL or OpenBLAS for BLAS backend support.
3. Build the project using the following commands:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### CMake Configuration

SQUINT provides several CMake options to customize the build:

- `-DSQUINT_BLAS_BACKEND`: Choose the BLAS backend (MKL, OpenBLAS, or NONE)
- `-DSQUINT_BUILD_DOCUMENTATION`: Build the documentation files (ON/OFF)
- `-DSQUINT_BUILD_TESTS`: Enable/disable building tests (ON/OFF)
- `-DCMAKE_BUILD_TYPE`: Set the build type (Debug, Release, etc.)

### BLAS Backends

SQUINT supports three BLAS backends to cater to different performance needs and system configurations:

1. Intel MKL: Optimized for high performance on Intel processors
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=MKL ..
   ```

2. OpenBLAS: An open-source alternative that's portable across different architectures
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=OpenBLAS ..
   ```

3. NONE: A limited fallback implementation for maximum portability
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=NONE ..
   ```

## Core Components

### Dimension System

The dimension system in SQUINT uses `std::ratio` for compile-time fraction representation of physical dimensions. Each dimension is represented by seven `std::ratio` values, corresponding to the seven SI base units:

1. Length (L)
2. Time (T)
3. Mass (M)
4. Temperature (K)
5. Electric Current (I)
6. Amount of Substance (N)
7. Luminous Intensity (J)

This system allows for precise and type-safe representation of complex physical dimensions. For example:

```cpp
using L = dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using T = dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using M = dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

using velocity_dim = dim_div_t<L, T>;
using acceleration_dim = dim_div_t<velocity_dim, T>;
using force_dim = dim_mult_t<M, acceleration_dim>;
```

### Quantity System

The `quantity` class template is the core of SQUINT's quantity system, representing values with associated dimensions:

```cpp
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
class quantity;
```

Where:
- `T` is the underlying arithmetic type (e.g., `float`, `double`, `int`)
- `D` is the dimension type
- `ErrorChecking` is the error checking policy

SQUINT provides alias templates for common quantity types to enhance readability:

```cpp
template <typename T> using length_t = quantity<T, dimensions::L>;
template <typename T> using time_t = quantity<T, dimensions::T>;
template <typename T> using velocity_t = quantity<T, dimensions::velocity_dim>;
// ... more aliases for other quantities
```

Quantities in SQUINT have all the properties of built-in arithmetic types, allowing for intuitive usage in calculations:

```cpp
auto distance = length_t<double>::meters(5.0);
auto time = time_t<double>::seconds(2.0);
auto speed = velocity_t<double>(distance / time);

// Quantities can be used in mathematical functions
auto acceleration = length_t<double>::meters(9.81) / (time_t<double>::seconds(1) * time_t<double>::seconds(1));
```

Importantly, dimensionless quantities can be used interchangeably with built-in arithmetic types, providing a seamless integration with existing code:

```cpp
auto dimensionless_value = distance / distance;  // Results in a dimensionless quantity
double scalar_value = 2.0 * dimensionless_value;  // No explicit conversion needed
```

All mathematical operations on quantities are performed in terms of the base unit. Conversion to/from specific units is handled during construction from non-quantity types or through explicit calls to the `unit_value` method.

#### Basic Operations

SQUINT provides a comprehensive set of mathematical operations for quantities:

- **Absolute Value**:
  ```cpp
  auto abs_value = abs(quantity);
  ```

- **Square Root**:
  ```cpp
  auto sqrt_value = sqrt(quantity);
  ```

- **Nth Root**:
  ```cpp
  auto nth_root = root<N>(quantity);
  ```

- **Exponential** (for dimensionless quantities):
  ```cpp
  auto exp_value = exp(dimensionless_quantity);
  ```

- **Logarithm** (for dimensionless quantities):
  ```cpp
  auto log_value = log(dimensionless_quantity);
  ```

- **Power**:
  ```cpp
  auto powered_value = pow<N>(quantity);
  ```

#### Trigonometric Functions

For dimensionless quantities, SQUINT provides standard trigonometric functions:

- **Sine, Cosine, Tangent**:
  ```cpp
  auto sin_value = sin(angle);
  auto cos_value = cos(angle);
  auto tan_value = tan(angle);
  ```

- **Inverse Trigonometric Functions**:
  ```cpp
  auto asin_value = asin(dimensionless_quantity);
  auto acos_value = acos(dimensionless_quantity);
  auto atan_value = atan(dimensionless_quantity);
  ```

- **Two-argument Arctangent**:
  ```cpp
  auto atan2_value = atan2(y, x);
  ```

#### Hyperbolic Functions

SQUINT also includes hyperbolic functions for dimensionless quantities:

- **Hyperbolic Sine, Cosine, Tangent**:
  ```cpp
  auto sinh_value = sinh(dimensionless_quantity);
  auto cosh_value = cosh(dimensionless_quantity);
  auto tanh_value = tanh(dimensionless_quantity);
  ```

- **Inverse Hyperbolic Functions**:
  ```cpp
  auto asinh_value = asinh(dimensionless_quantity);
  auto acosh_value = acosh(dimensionless_quantity);
  auto atanh_value = atanh(dimensionless_quantity);
  ```

#### Comparison

SQUINT provides an approximate equality function for comparing quantities:

- **Approximate Equality**:
  ```cpp
  bool are_equal = approx_equal(quantity1, quantity2, epsilon);
  ```

### Tensor System

SQUINT's tensor system is built around a single, flexible `tensor` class with a policy-based design, supporting both fixed and dynamic shapes:

```cpp
template <typename T, typename Shape, typename Strides = strides::column_major<Shape>,
          error_checking ErrorChecking = error_checking::disabled,
          ownership_type OwnershipType = ownership_type::owner,
          memory_space MemorySpace = memory_space::host>
class tensor;
```

Key features of the tensor system include:
- Single class design for both fixed and dynamic shapes
- Compile-time optimizations for fixed shapes
- Runtime flexibility for dynamic shapes
- Configurable error checking
- Flexible memory ownership (owner or reference)
- Support for different memory spaces
- Column-major default layout for construction and iteration

The library includes aliases for common tensor types to improve code readability:

```cpp
template <typename T> using vec3_t = tensor<T, shape<3>>;
template <typename T> using mat3_t = tensor<T, shape<3, 3>>;
// ... more aliases for other tensor types

using vec3 = vec3_t<float>;
using mat3 = mat3_t<float>;
// ... more type-specific aliases
```

## Usage

### Quantity Usage

Quantities in SQUINT can be used in a variety of ways, showcasing their flexibility and integration with both scalar and tensor operations:

```cpp
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
```

### Tensor Construction

SQUINT provides several ways to construct tensors, with a default column-major layout:

1. Using initializer lists (column-major order):

```cpp
mat2x3 A{1, 4, 2, 5, 3, 6};
// A represents:
// [1 2 3]
// [4 5 6]
```

2. Factory methods:

```cpp
auto zero_matrix = mat3::zeros();
auto ones_matrix = mat4::ones();
auto identity_matrix = mat3::eye();
auto random_matrix = mat3::random(0.0, 1.0);
```

3. Element-wise initialization:

```cpp
mat3 custom_matrix;
for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
        custom_matrix(i, j) = i * 3 + j;  // Note the use of () for element access
    }
}
```

4. Construction from other tensors or views:

```cpp
mat3 original{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
mat3 copy(original);

mat4 big_matrix = mat4::random(0.0, 1.0);
mat3 sub_matrix(big_matrix.subview<3, 3>(0, 0));
```

5. Dynamic tensor construction:

```cpp
std::vector<size_t> shape = {3, 4, 5};
dynamic_tensor<float> dynamic_tensor(shape);
dynamic_tensor<float> filled_tensor(shape, 1.0f);
```

6. Tensor construction with quantities:

```cpp
vec3_t<length_t<double>> position{
    length_t<double>::meters(1.0),
    length_t<double>::meters(2.0),
    length_t<double>::meters(3.0)
};
```

### Basic Operations

SQUINT supports a wide range of operations for tensors:

```cpp
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
```

### Views and Reshaping

SQUINT provides powerful view and reshaping capabilities:

```cpp
auto view = A.view();  // Create a view of the entire tensor
auto subview = A.subview<2, 2>(0, 1);  // Create a 2x2 subview starting at (0, 1)
auto reshaped = A.reshape<6>();  // Reshape to a 1
D tensor
auto transposed = A.transpose();  // Transpose the tensor
auto permuted = A.permute<1,0>(); // Permutation of the tensor

// For dynamic tensors
auto dynamic_reshaped = dynamic_tensor.reshape({6, 4});
auto dynamic_transposed = dynamic_tensor.transpose();
```

### Linear Algebra Operations

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

### Vector Operations

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

### Matrix Operations

SQUINT provides essential matrix operations:

- **Trace**:
  ```cpp
  auto matrix_trace = trace(A);
  ```

### Statistical Functions

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

### Comparison

SQUINT provides an approximate equality function for comparing tensors:

- **Approximate Equality**:
  ```cpp
  bool are_equal = approx_equal(A, B, tolerance);
  ```

### Tensor Contraction (for dynamic tensors)

SQUINT supports tensor contraction operations:

- **Tensor Contraction**:
  ```cpp
  auto contracted = contract(A, B, contraction_pairs);
  ```

## Advanced Features

### Error Checking

SQUINT provides optional runtime error checking for both quantities and tensors. This feature can be enabled or disabled at compile-time:

```cpp
using checked_quantity = quantity<double, length, error_checking::enabled>;
tensor<float, shape<3, 3>, strides::column_major<shape<3, 3>>, error_checking::enabled> checked_tensor;
```

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

### Unit Conversions

SQUINT supports seamless unit conversions within the same dimension:

```cpp
auto meters = length_t<double>::meters(5.0);
auto feet = convert_to<units::feet_t>(meters);
auto inches = convert_to<units::inches_t>(meters);

// Unit conversions can also be done directly in calculations
auto speed = length_t<double>::kilometers(60.0) / time_t<double>::hours(1.0);
auto speed_mph = convert_to<units::miles_per_hour_t>(speed);
```

### Constants

SQUINT includes a comprehensive set of physical and mathematical constants:

```cpp
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
```

These constants are implemented as `constant_quantity_t` types, ensuring proper dimensional analysis in calculations.

## API Reference

For a complete API reference, please refer to the inline documentation in the header files. The documentation provides detailed information about each class, function, and template, including:

- Template parameters and their constraints
- Function parameters and return types
- Preconditions and postconditions
- Exception specifications
- Usage examples

## Performance Considerations

To get the best performance out of SQUINT:

1. Use fixed-size tensors when dimensions are known at compile-time. This allows for more aggressive compiler optimizations.

2. Choose the appropriate BLAS backend for your hardware and use case:
   - Use Intel MKL on Intel processors for best performance
   - Use OpenBLAS for good performance on a variety of architectures
   - Use the NONE backend only when portability is the highest priority

3. Prefer views over copies for subsections of tensors to avoid unnecessary memory allocations and copies.

4. Disable error checking in performance-critical code paths once you're confident in your implementation's correctness.
