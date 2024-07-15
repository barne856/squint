# SQUINT C++ Library

## Table of Contents

1. [Introduction](#introduction)
2. [Design Principles](#design-principles)
3. [Core Components](#core-components)
   - [Dimension System](#dimension-system)
   - [Quantity System](#quantity-system)
   - [Tensor System](#tensor-system)
   - [Units and Constants](#units-and-constants)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Basic Quantity Operations](#basic-quantity-operations)
   - [Using Physical Constants](#using-physical-constants)
   - [Basic Tensor Operations](#basic-tensor-operations)
   - [Linear Algebra Operations](#linear-algebra-operations)
6. [Advanced Usage](#advanced-usage)
   - [Working with Mixed Types](#working-with-mixed-types)
   - [Combining Quantities and Tensors](#combining-quantities-and-tensors)
   - [Tensor Views and Reshaping](#tensor-views-and-reshaping)
7. [API Reference](#api-reference)
   - [Dimension System](#dimension-system-1)
   - [Quantity System](#quantity-system-1)
   - [Tensor System](#tensor-system-1)
   - [Units Namespace](#units-namespace)
   - [Constants Namespace](#constants-namespace)
   - [Math Namespace](#math-namespace)
   - [Linear Algebra Operations](#linear-algebra-operations-1)
8. [Building and Testing](#building-and-testing)
   - [On Windows](#on-windows)
   - [On Linux](#on-linux)
   - [Running Tests](#running-tests)

## Introduction

SQUINT (Static Quantities in Tensors) is a modern, header-only C++ library designed to bring robust dimensional analysis, unit conversion, and linear algebra operations to C++. By leveraging C++'s template metaprogramming capabilities, Squint provides a powerful set of tools that enhance code safety, readability, and expressiveness without compromising performance.

Key features of Squint include:

1. **Dimensional Analysis**: Squint implements a compile-time dimensional analysis system that catches unit-related errors at compile-time, preventing common mistakes in scientific and engineering calculations.
2. **Unit Conversions**: The library offers a wide range of predefined units and easy-to-use conversion functions, simplifying the often error-prone process of unit conversion.
3. **Quantity System**: Squint's quantity system allows for intuitive representation of physical quantities, complete with unit information, enabling natural arithmetic operations between quantities of compatible dimensions.
4. **Tensor Operations**: The library provides both fixed-size and dynamic-size tensor classes, supporting various linear algebra operations, element-wise arithmetic, and advanced indexing, views, and slicing capabilities.
5. **Error Checking**: Optional runtime error checking for operations like overflow, underflow, division by zero, compatible matrix shapes for operations, and bounds checking adds an extra layer of safety to numerical computations.
6. **Constants and Common Functions**: Squint includes a comprehensive set of mathematical and physical constants, as well as functions commonly used in scientific computing.
7. **Performance**: Despite its high-level abstractions, Squint is designed with performance in mind, utilizing template metaprogramming techniques to minimize or eliminate runtime overhead.

## Design Principles

Squint is built on several core principles that guide its design and implementation:

1. **Type Safety**: Squint leverages C++'s strong type system to catch errors at compile-time rather than runtime.
2. **Expressiveness**: The library aims to allow users to write code that closely mirrors mathematical and physical equations.
3. **Flexibility**: Squint is designed to be flexible, allowing users to work with different numeric types and providing both fixed-size and dynamic-size tensors.
4. **Performance**: Squint uses template metaprogramming techniques to minimize or eliminate runtime overhead.
5. **Extensibility**: The library is designed to be extensible, allowing users to define their own units, dimensions, and potentially even custom tensor types.

## Core Components

### Dimension System

The dimension system in Squint is implemented using `std::ratio` for compile-time fraction representation. It includes base dimensions such as length, time, mass, temperature, current, amount of substance, and luminous intensity. Compound dimensions are created through template metaprogramming.

```cpp
// Example of base and compound dimensions
using length = dimension<std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using time = dimension<std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
using velocity = div_t<length, time>;
```

### Quantity System

The `quantity` class template is the core of the Squint library. It represents a value with an associated dimension and supports various arithmetic operations and unit conversions.

```cpp
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
class quantity;
```

### Tensor System

Squint provides both fixed-size and dynamic-size tensor types:

1. `fixed_tensor<T, L, ErrorChecking, Dims...>`: For tensors whose dimensions are known at compile-time
2. `dynamic_tensor<T, ErrorChecking>`: For tensors whose dimensions are determined at runtime

These tensor types support multi-dimensional indexing, slicing, views, and various linear algebra operations.

#### BLAS and LAPACK Dependencies

The SQUINT tensor system relies on efficient implementations of Basic Linear Algebra Subprograms (BLAS) and Linear Algebra Package (LAPACK) for high-performance linear algebra operations. These libraries provide optimized routines for operations such as matrix multiplication, solving linear systems, eigenvalue computations, and more.

SQUINT allows users to choose between different BLAS and LAPACK implementations through the `BLAS_BACKEND` CMake option. Currently, SQUINT supports two backends:

1. **Intel Math Kernel Library (MKL)**: A highly optimized mathematical library developed by Intel, which includes implementations of BLAS and LAPACK. MKL is particularly well-suited for Intel processors and can provide significant performance benefits on compatible hardware.

2. **OpenBLAS**: An open-source optimized BLAS library that also includes a subset of LAPACK routines. OpenBLAS is portable across different architectures and provides good performance on a wide range of systems.

#### Configuring the BLAS Backend

To configure the BLAS backend, use the `BLAS_BACKEND` CMake option when building SQUINT. Here's how to set it for each supported backend:

1. For Intel MKL:
   ```
   cmake -DBLAS_BACKEND=MKL ..
   ```

2. For OpenBLAS:
   ```
   cmake -DBLAS_BACKEND=OpenBLAS ..
   ```

The choice of backend can significantly impact performance depending on your hardware and specific use case. Users are encouraged to experiment with both options to determine which provides the best performance for their particular setup.

#### Impact on Tensor Operations

The choice of BLAS backend affects the performance of various tensor operations, particularly for large-scale computations. Operations that benefit from the BLAS/LAPACK implementation include:

- Matrix multiplication
- Matrix-vector multiplication
- Solving linear systems

For smaller tensors or simpler operations, the impact of the BLAS backend choice may be less noticeable, as SQUINT can optimize these cases directly.

#### Extensibility

The tensor system in SQUINT is designed with extensibility in mind. While currently supporting MKL and OpenBLAS, the architecture allows for relatively easy integration of other BLAS/LAPACK implementations in the future. This could include GPU-accelerated libraries or other specialized implementations for specific hardware or use cases.

### Units and Constants

Squint includes a comprehensive set of predefined units for common physical quantities and mathematical and physical constants. These are organized in the `units` and `constants` namespaces, respectively.

## Installation

Squint is a header-only library. To use it in your project:

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

## Usage

### Basic Quantity Operations

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

### Using Physical Constants

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

### Basic Tensor Operations

```cpp
#include <squint/tensor.hpp>
#include <iostream>

int main() {
    using namespace squint;

    // Create a 2x3 fixed tensor (flat array in column major order)
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

### Linear Algebra Operations

```cpp
#include <squint/tensor.hpp>
#include <iostream>

int main() {
    using namespace squint;

    mat3 a{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    mat3 b{{9, 8, 7, 6, 5, 4, 3, 2, 1}};

    // Matrix addition
    auto sum = a + b;

    // Matrix multiplication
    auto product = a * b;

    // Matrix transposition
    auto transposed = a.transpose();

    // Matrix inversion
    auto inverted = a.inv();

    // Solve linear system
    vec3 x{{1, 2, 3}};
    auto solution = solve(a, x);

    return 0;
}
```

## Advanced Usage

### Working with Mixed Types

Squint supports operations between quantities with different underlying types:

```cpp
quantity<double, length> l_double(5.0);
quantity<float, length> l_float(3.0F);
quantity<int, length> l_int(2);

auto result = l_double + l_float + l_int; // result is quantity<double, length>
```

### Combining Quantities and Tensors

One of the powerful features of Squint is the ability to create tensors of quantities, allowing for type-safe multi-dimensional physical calculations.

```cpp
// A 3D vector of times
vec3_t<time> times = {
    time::seconds(1.0),
    time::seconds(2.0),
    time::seconds(3.0)
};

// A 3x3 matrix of velocities
mat3_t<velocity> velocity_field = {
    velocity::meters_per_second(1.0), velocity::meters_per_second(2.0), velocity::meters_per_second(3.0),
    velocity::meters_per_second(4.0), velocity::meters_per_second(5.0), velocity::meters_per_second(6.0),
    velocity::meters_per_second(7.0), velocity::meters_per_second(8.0), velocity::meters_per_second(9.0)
};

// Perform operations
auto lengths = velocity_field * times; // Results in a vec3_t<quantity<double, length>>
```

### Tensor Views and Reshaping

```cpp
mat3x4_t<float> tensor{{/* ... */}};

// Create a view
auto view = tensor.view();

// Create a subview
auto subview = tensor.subview<2, 2>(slice{0, 2}, slice{1, 3});

// Reshape tensor
auto reshaped = tensor.reshape<6, 2>();

// Flatten tensor
auto flattened = tensor.flatten();
```

## API Reference

(The API Reference section remains unchanged from the original README)

## Building and Testing

### On Windows

1. Install MSVC, CMake >= 3.28, Clang >= 15, and optionally MKL and Intel Fortran compiler.
2. Set up environment variables:
   ```
   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
   ```
3. Build the project with OpenBLAS:
   ```
   mkdir build
   cd build
   cmake -DCMAKE_Fortran_COMPILER=ifx -DBLAS_BACKEND=OpenBLAS -T ClangCL ..
   cd ..
   cmake --build ./build
   ```
4. OR - Build the project with MKL (no Fortran compiler needed)
   ```
   mkdir build
   cd build
   cmake -DBLAS_BACKEND=MKL -T ClangCL ..
   cd ..
   cmake --build ./build
   ```

### On Linux

1. Install CMake >= 3.28, Clang >= 15, and optionally MKL and gfortran.
2. Build the project with OpenBLAS:
   ```
   mkdir build
   cd build
   cmake -DBLAS_BACKEND=OpenBLAS ..
   cd ..
   cmake --build ./build
   ```
3. OR - Build the project with MKL (no Fortran compiler needed)
   ```
   mkdir build
   cd build
   cmake -DBLAS_BACKEND=MKL ..
   cd ..
   cmake --build ./build
   ```

### Running Tests

```
cd build
ctest
```
