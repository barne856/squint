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

# SQUINT Tensor Library

## Introduction

SQUINT (Static Quantities in Tensors) is a header-only C++ library designed for compile-time dimensional analysis, unit conversion, and linear algebra operations. It combines a quantity system for handling physical units and dimensions with a tensor system for numerical computations.

SQUINT was developed primarily to suit my personal needs and preferences. It is not designed to be the fastest or most straightforward tensor library available. Instead, it prioritizes type safety, expressiveness, and a cohesive API that integrates well with physical simulations and graphics programming.

The primary goals of SQUINT are:

1. To provide a type-safe framework for calculations involving physical quantities, catching dimension-related errors at compile-time where possible.
2. To offer a tensor system with an API that balances ease of use with static type checking.
3. To integrate seamlessly with physical quantities, enabling tensor operations on dimensioned values.
4. To make an honest effort at *good* performance.

SQUINT is particularly suited for projects where type safety and dimensional correctness are important, such as physics engines, scientific simulations, or graphics applications dealing with real-world units. It aims to catch errors early in the development process while providing a comfortable API for both quantities and tensors.

While the library makes efforts to be performant, especially through the use of compile-time optimizations and BLAS integration, users requiring absolute peak performance or a minimalist API might find other specialized libraries more suitable for their needs.

## Key Features

Key features of SQUINT include:

- Compile-time dimensional analysis
- A flexible tensor system supporting both fixed and dynamic shapes
- Integration of physical quantities with tensor operations
- Optional runtime error checking
- Support for common linear algebra operations
- Useful mathematical and physical constants

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

.. note::
  MSVC is partially supported but lacks support for multidimensional subscript operators.

### Build Instructions

1. Ensure you have CMake version 3.28 or later and a supported compiler installed.
2. Optionally install MKL if you intend to use it as a BLAS backend.
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

.. note::
  For the OpenBLAS backend, SQUINT will automatically fetch the source code from github and build it from source along with the library if you use the provided CMakeLists.txt file.

### Serving Documentation

If SQUINT was built with documentation, you can serve it locally using

```
python -m http.server -d ./build/sphinx
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

New dimensions can be created by combining these base dimensions. For example, the dimension of force (F) can be represented as:

$F = M \cdot L \cdot T^{-2}$

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

Quantities in SQUINT have all the properties of built-in arithmetic types, allowing for intuitive usage in calculations. For instance, calculating acceleration:

$a = \frac{d}{t^2}$

```cpp
// Quantities can be used in mathematical functions
auto acceleration = length_t<double>::meters(9.81) / (time_t<double>::seconds(1) * time_t<double>::seconds(1));
```

Importantly, dimensionless quantities can be used interchangeably with built-in arithmetic types, providing a seamless integration with existing code:

```cpp
auto dimensionless_value = distance / distance;  // Results in a dimensionless quantity
double scalar_value = 2.0 * dimensionless_value;  // No explicit conversion needed
```

All mathematical operations on quantities are performed in terms of the base unit. Conversion to/from specific units is handled during construction from non-quantity types or through explicit calls to the `unit_value` method.

It is important to note that the size of any quantity type is exactally the size of it's underlying arithmetic type. This can be important for some applications. For example:

```
sizeof(float) == sizeof(length_t<float>);
```

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

For dimensionless quantities, SQUINT provides standard trigonometric functions for dimensionless quantities:

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

In addition to the standard comparision operators, SQUINT provides an approximate equality function for comparing quantities:

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

It is important to note that the size of any fixed shape tensor type is exactally the size of it's elements. This can be important for some applications. For example:

```
sizeof(vec3) == 3 * sizeof(float);
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
// and more ...
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

For matrix multiplication, the operation performed is:

$(AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$

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

SQUINT provides comprehensive linear algebra operations:

- **Solving Linear Systems**:

```cpp
auto result = solve(A, b);  // Solves Ax = b for square systems
```

This solves the system of linear equations:
  
$Ax = b$

- **Matrix Inversion**:

```cpp
auto inverse = inv(A);  // Computes the inverse of a square matrix
```

The inverse $A^{-1}$ satisfies:
  
$AA^{-1} = A^{-1}A = I$

- **Pseudoinverse**:

```cpp
auto pseudo_inverse = pinv(A);  // Computes the Moore-Penrose pseudoinverse
```

For a matrix $A$, the Moore-Penrose pseudoinverse $A^+$ satisfies:
  
$AA^+A = A$
$A^+AA^+ = A^+$
$(AA^+)^* = AA^+$
$(A^+A)^* = A^+A$

### Vector Operations

- **Cross Product** (for 3D vectors):

```cpp
auto cross_product = cross(a, b);
```

For vectors $a = (a_x, a_y, a_z)$ and $b = (b_x, b_y, b_z)$:
  
$a \times b = (a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x)$

- **Dot Product**:

```cpp
auto dot_product = dot(a, b);
```

For vectors $a$ and $b$:
  
$a \cdot b = \sum_{i=1}^n a_i b_i$

- **Vector Norm**:

```cpp
auto vector_norm = norm(a);
```

The Euclidean norm of a vector $a$ is:
  
$\|a\| = \sqrt{\sum_{i=1}^n |a_i|^2}$

### Matrix Operations

- **Trace**:

```cpp
auto matrix_trace = trace(A);
```

The trace of a square matrix $A$ is:
  
$\text{tr}(A) = \sum_{i=1}^n A_{ii}$

### Statistical Functions

- **Mean**:

```cpp
auto tensor_mean = mean(A);
```

For a tensor $A$ with $n$ elements:
  
$\text{mean}(A) = \frac{1}{n} \sum_{i=1}^n A_i$

### Tensor Contraction (for dynamic tensors)

- **Tensor Contraction**:

```cpp
auto contracted = contract(A, B, contraction_pairs);
```

For tensors $A$ and $B$, the contraction over indices $i$ and $j$ is:
  
$(A \cdot B)_{k_1...k_n l_1...l_m} = \sum_{i,j} A_{k_1...k_n i} B_{j l_1...l_m}$

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

SQUINT includes a comprehensive set of physical and mathematical constants for example:

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

These constants and more are implemented as `constant_quantity_t` types, ensuring proper dimensional analysis in calculations.

### Tensor Views with Step Sizes

SQUINT supports creating tensor views with custom step sizes for both fixed and dynamic shape tensors, allowing for more flexible and efficient data access patterns. This feature is particularly useful for operations like strided slicing or accessing every nth element along a dimension.

#### API for Fixed Shape Tensors

For fixed shape tensors, SQUINT provides a template method that determines the step sizes at compile-time:

```cpp
template <typename SubviewShape, typename StepSizes>
auto subview(const index_type& start_indices);
```

Here, `SubviewShape` defines the shape of the resulting view, and `StepSizes` defines the step sizes along each dimension.

Usage example for fixed shape tensors:

```cpp
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
```

#### API for Dynamic Shape Tensors

For dynamic shape tensors, SQUINT provides a method that takes runtime arguments for the subview shape, start indices, and step sizes:

```cpp
auto subview(const index_type& subview_shape, const index_type& start_indices, const index_type& step_sizes);
```

Usage example for dynamic shape tensors:

```cpp
// Create a 10x10x10 tensor
dynamic_tensor<float> B({10, 10, 10});

// Create a 3x3x3 view with elements spaced 3 apart in each dimension
auto custom_view = B.subview({3, 3, 3}, {1, 1, 1}, {3, 3, 3});
```

When using views with step sizes, keep in mind:

- The resulting view is not guaranteed to be contiguous in memory.
- Operations on these views may be less efficient than on contiguous data, depending on the hardware and BLAS backend.
- For fixed shape tensors, the shape and step sizes are checked at compile-time, providing additional type safety.
- For dynamic shape tensors, the shape of the resulting view is determined by the `subview_shape` parameter, not by the original tensor's shape and the step sizes.

## API Reference

A complete API reference is included with this documentation, you can also refer to the inline documentation in the header files which is used to generate the API reference.

## Performance Considerations

To get the best performance out of SQUINT:

1. Use fixed-size tensors when dimensions are known at compile-time. This allows for more aggressive compiler optimizations.

2. Choose the appropriate BLAS backend for your hardware and use case:
   - Use Intel MKL on Intel processors for best performance
   - Use OpenBLAS for good performance on a variety of architectures
   - Use the NONE backend only when portability is the highest priority

3. Prefer views over copies for subsections of tensors to avoid unnecessary memory allocations and copies.

4. Disable error checking in performance-critical code paths once you're confident in your implementation's correctness.
