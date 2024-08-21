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

## Key Features

- Compile-time dimensional analysis
- Efficient tensor operations with both fixed and dynamic shapes
- Support for physical quantities with units
- Flexible error checking policies
- Integration with BLAS and LAPACK for high-performance linear algebra
- Header-only design for easy integration

## Installation

SQUINT is a header-only library. To use it in your project:

1. Copy the `include/squint` directory to your project's include path.
2. Include the necessary headers in your C++ files:

```cpp
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>
```

For CMake projects, you can use FetchContent:

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

SQUINT requires a C++23 compliant compiler. Currently supported compilers:

- GCC (g++) version 12 or later
- Clang version 15 or later

Note: MSVC is partially supported but lacks multidimensional subscript operators.

### Build Instructions

1. Install CMake >= 3.28 and a supported compiler.
2. Optionally install MKL or OpenBLAS for BLAS backend support.
3. Build the project:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### CMake Configuration

Key CMake options:

- `-DSQUINT_BLAS_BACKEND`: Choose the BLAS backend (MKL, OpenBLAS, or NONE)
- `-DSQUINT_BUILD_DOCUMENTATION`: Build the documentation files (ON/OFF)
- `-DSQUINT_BUILD_TESTS`: Enable/disable building tests (ON/OFF)
- `-DCMAKE_BUILD_TYPE`: Set the build type (Debug, Release, etc.)

### BLAS Backends

SQUINT supports three BLAS backends:

1. Intel MKL: High performance on Intel processors
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=MKL ..
   ```

2. OpenBLAS: Open-source, portable across architectures
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=OpenBLAS ..
   ```

3. NONE: Limited fallback implementation for portability
   ```bash
   cmake -DSQUINT_BLAS_BACKEND=NONE ..
   ```

## Core Components

### Dimension System

The dimension system uses `std::ratio` for compile-time fraction representation of physical dimensions. Each dimension is represented by seven `std::ratio` values, corresponding to the seven SI base units:

1. Length (L)
2. Time (T)
3. Mass (M)
4. Temperature (K)
5. Electric Current (I)
6. Amount of Substance (N)
7. Luminous Intensity (J)

Example:

```cpp
using L = dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using T = dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using M = dimension<std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

using velocity_dim = dim_div_t<L, T>;
using acceleration_dim = dim_div_t<velocity_dim, T>;
using force_dim = dim_mult_t<M, acceleration_dim>;
```

### Quantity System

The `quantity` class template represents values with associated dimensions:

```cpp
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
class quantity;
```

Where:
- `T` is the underlying arithmetic type (e.g., `float`, `double`, `int`)
- `D` is the dimension type
- `ErrorChecking` is the error checking policy

SQUINT provides alias templates for common quantity types:

```cpp
template <typename T> using length_t = quantity<T, dimensions::L>;
template <typename T> using time_t = quantity<T, dimensions::T>;
template <typename T> using velocity_t = quantity<T, dimensions::velocity_dim>;
// ... more aliases for other quantities
```

Usage example:

```cpp
auto distance = length_t<double>::meters(5.0);
auto time = time_t<double>::seconds(2.0);
auto speed = velocity_t<double>(distance / time);
```

All mathematical operations on quantities are performed in terms of the base unit. Conversion to/from units is handled during construction from non-quantity types or through explicit calls to the `unit_value` method.

#### Basic Operations

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

- **Exponential**:
  ```cpp
  auto exp_value = exp(dimensionless_quantity);
  ```

- **Logarithm**:
  ```cpp
  auto log_value = log(dimensionless_quantity);
  ```

- **Power**:
  ```cpp
  auto powered_value = pow<N>(quantity);
  ```

#### Trigonometric Functions

For dimensionless quantities:

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

For dimensionless quantities:

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

- **Approximate Equality**:
  ```cpp
  bool are_equal = approx_equal(quantity1, quantity2, epsilon);
  ```

### Tensor System

SQUINT uses a single, flexible `tensor` class with a policy-based design, supporting both fixed and dynamic shapes:

```cpp
template <typename T, typename Shape, typename Strides = strides::column_major<Shape>,
          error_checking ErrorChecking = error_checking::disabled,
          ownership_type OwnershipType = ownership_type::owner,
          memory_space MemorySpace = memory_space::host>
class tensor;
```

Key features:
- Single class design for fixed and dynamic shapes
- Compile-time optimizations for fixed shapes
- Runtime flexibility for dynamic shapes
- Configurable error checking
- Flexible memory ownership (owner or reference)
- Support for different memory spaces

The library includes aliases for common tensor types:

```cpp
template <typename T> using vec3_t = tensor<T, shape<3>>;
template <typename T> using mat3_t = tensor<T, shape<3, 3>>;
// ... more aliases for other tensor types

using vec3 = vec3_t<float>;
using mat3 = mat3_t<float>;
// ... more type-specific aliases
```

## Usage

### Tensor Construction

SQUINT provides several ways to construct tensors:

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
        custom_matrix[i, j] = i * 3 + j;
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

```cpp
auto C = A + B;  // Element-wise addition
auto D = A * B;  // Matrix multiplication
auto E = A * 2.0;  // Scalar multiplication
```

### Views and Reshaping

```cpp
auto view = A.view();  // Create a view of the entire tensor
auto subview = A.subview<2, 2>(0, 1);  // Create a 2x2 subview starting at (0, 1)
auto reshaped = A.reshape<6>();  // Reshape to a 1D tensor
```

### Linear Algebra Operations

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

- **Trace**:
  ```cpp
  auto matrix_trace = trace(A);
  ```

### Statistical Functions

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

- **Approximate Equality**:
  ```cpp
  bool are_equal = approx_equal(A, B, tolerance);
  ```

### Tensor Contraction (for dynamic tensors)

- **Tensor Contraction**:
  ```cpp
  auto contracted = contract(A, B, contraction_pairs);
  ```

## Advanced Features

### Error Checking

SQUINT provides optional runtime error checking:

```cpp
using checked_quantity = quantity<double, length, error_checking::enabled>;
tensor<float, shape<3, 3>, strides::column_major<shape<3, 3>>, error_checking::enabled> checked_tensor;
```

### Unit Conversions

```cpp
auto meters = length_t<double>::meters(5.0);
auto feet = convert_to<units::feet_t>(meters);
```

### Constants

SQUINT includes various physical and mathematical constants:

```cpp
auto c = si_constants<double>::c;  // Speed of light
auto pi = math_constants<double>::pi;  // Pi
```

## API Reference

For a complete API reference, refer to the inline documentation in the header files.

## Performance Considerations

- Use fixed-size tensors when dimensions are known at compile-time for better optimization.
- Choose the appropriate BLAS backend for your hardware and use case.
- Prefer views over copies for subsections of tensors.

## Troubleshooting

- Ensure you're using a supported compiler version.
- Check that the BLAS backend is correctly configured.
- For dimension-related errors, verify that quantities have compatible dimensions.
- When using MSVC, be aware of the limitations regarding multidimensional subscript operators.

For more detailed information on specific components or usage patterns, refer to the inline documentation and examples in the source code.