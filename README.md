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

### Dimension System

The dimension system is implemented using `std::ratio` for compile-time fraction representation.

#### Base Dimensions

- `squint::dimensions::length`
- `squint::dimensions::time`
- `squint::dimensions::mass`
- `squint::dimensions::temperature`
- `squint::dimensions::current`
- `squint::dimensions::amount_of_substance`
- `squint::dimensions::luminous_intensity`


#### Compound Dimensions

Compound dimensions are created through template metaprogramming. For example:

```cpp
using velocity = dim_div<length, time>;
using acceleration = dim_div<velocity, time>;
using force = dim_mult<mass, acceleration>;
```

### Quantity System

The `quantity` class template is the core of the SQUINT library.

```cpp
template <typename T, typename D, error_checking ErrorChecking = error_checking::disabled>
class quantity;
```

#### Template Parameters

- `T`: The underlying arithmetic type (e.g., `float`, `double`, `int`)
- `D`: The dimension type
- `ErrorChecking`: Error checking policy (`error_checking::enabled` or `error_checking::disabled`)

#### Constructors

```cpp
quantity();  // Default constructor, initializes to T{}
explicit quantity(const T& value);  // Constructs from value
quantity(const U& value);  // Implicit conversion for dimensionless quantities
```

#### Public Methods

```cpp
constexpr T value() const noexcept;  // Returns underlying value
template <template <typename, typename> typename TargetUnit, typename TargetErrorChecking = ErrorChecking>
constexpr auto as() const;  // Converts to different unit
template <int N>
constexpr auto pow() const;  // Raises quantity to power N
template <int N>
auto root() const;  // Takes Nth root
constexpr auto sqrt() const;  // Square root
```

#### Operators

- Arithmetic operators: `+`, `-`, `*`, `/`
- Comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`

### Tensor System

#### Fixed Tensor

```cpp
template <typename T, layout L, error_checking ErrorChecking, size_t... Dims>
class fixed_tensor;
```

##### Template Parameters

- `T`: The underlying data type
- `L`: Memory layout (`layout::row_major` or `layout::column_major`)
- `ErrorChecking`: Error checking policy
- `Dims...`: Compile-time dimensions of the tensor

##### Constructors

```cpp
fixed_tensor();  // Default constructor
fixed_tensor(const std::array<T, total_size>& elements);  // Constructs from flat array
explicit fixed_tensor(const T& value);  // Fills tensor with single value
fixed_tensor(const BlockTensor& block);  // Constructs from smaller tensor block
```

##### Public Methods

```cpp
constexpr size_t rank() const noexcept;  // Returns tensor rank
constexpr size_t size() const noexcept;  // Returns total number of elements
constexpr auto shape() const noexcept;  // Returns shape of tensor
constexpr auto strides() const noexcept;  // Returns strides of tensor
T& at(size_t... indices);  // Element access with bounds checking
T& operator[](size_t... indices);  // Element access without bounds checking
T* data() noexcept;  // Returns pointer to underlying data
auto subview(Slices... slices);  // Creates subview of tensor
auto view();  // Creates view of entire tensor
template <size_t... NewDims>
auto reshape();  // Reshapes tensor
auto flatten();  // Returns flattened view of tensor
auto rows();  // Returns row views
auto cols();  // Returns column views
```

##### Static Methods

```cpp
static auto zeros();  // Creates tensor filled with zeros
static auto ones();  // Creates tensor filled with ones
static auto full(const T& value);  // Creates tensor filled with specific value
static auto random(T min, T max);  // Creates tensor with random values
static auto I();  // Creates identity tensor
static auto diag(const T& value);  // Creates diagonal tensor
```

#### Dynamic Tensor

```cpp
template <typename T, error_checking ErrorChecking = error_checking::disabled>
class dynamic_tensor;
```

##### Template Parameters

- `T`: The underlying data type
- `ErrorChecking`: Error checking policy

##### Constructors

```cpp
dynamic_tensor();  // Default constructor
dynamic_tensor(std::vector<size_t> shape, layout l = layout::column_major);  // Constructs with given shape
dynamic_tensor(std::vector<size_t> shape, const std::vector<T>& elements, layout l = layout::column_major);  // Constructs from elements
```

##### Public Methods

Similar to `fixed_tensor`, plus:

```cpp
void reshape(std::vector<size_t> new_shape);  // Reshape tensor in-place
```

#### Tensor Views

```cpp
template <typename T, layout L, error_checking ErrorChecking, size_t... Dims>
class fixed_tensor_view;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
class dynamic_tensor_view;
```

These provide non-owning views into tensor data, with similar methods to their owning counterparts.

### Units Namespace

The `units` namespace contains various unit types and conversion functions. Some of the key unit types include:

```cpp
template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct length_t;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct time_t;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct mass_t;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct temperature_t;

template <typename T, error_checking ErrorChecking = error_checking::disabled>
struct velocity_t;

// ... and more
```

Each unit type provides static methods for creating quantities in different units. For example:

```cpp
auto len = length_t<double>::meters(5.0);
auto spd = velocity_t<double>::kilometers_per_hour(60.0);
```

### Constants Namespace

The `constants` namespace provides template structs with various mathematical and physical constants:

```cpp
template <typename T>
struct math_constants {
    static constexpr auto pi = quantity<T, dimensionless>(3.14159265358979323846);
    static constexpr auto e = quantity<T, dimensionless>(2.71828182845904523536);
    static constexpr auto sqrt2 = quantity<T, dimensionless>(1.41421356237309504880);
    // ... and more
};

template <typename T>
struct si_constants {
    static constexpr auto c = quantity<T, velocity>(299792458.0);  // Speed of light
    static constexpr auto h = quantity<T, energy_time>(6.62607015e-34);  // Planck constant
    static constexpr auto G = quantity<T, force_length_squared_per_mass_squared>(6.67430e-11);  // Gravitational constant
    // ... and more
};

template <typename T>
struct astro_constants {
    static constexpr auto AU = quantity<T, length>(1.495978707e11);  // Astronomical unit
    static constexpr auto parsec = quantity<T, length>(3.0856775814913673e16);
    static constexpr auto light_year = quantity<T, length>(9.4607304725808e15);
    // ... and more
};

template <typename T>
struct atomic_constants {
    static constexpr auto R_inf = quantity<T, wave_number>(10973731.568160);  // Rydberg constant
    static constexpr auto a_0 = quantity<T, length>(5.29177210903e-11);  // Bohr radius
    // ... and more
};
```

### Math Namespace

The `math` namespace provides overloads for common math functions to work with quantities:

```cpp
template <typename T, typename D>
quantity<T, D> abs(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dim_root<D, 2>> sqrt(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> exp(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> log(const quantity<T, D>& q);

template <typename T, typename D1, typename D2>
quantity<T, dim_mult<D1, D2>> pow(const quantity<T, D1>& base, const quantity<T, D2>& exponent);

// Trigonometric functions
template <typename T, typename D>
quantity<T, dimensionless> sin(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> cos(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> tan(const quantity<T, D>& q);

// Inverse trigonometric functions
template <typename T, typename D>
quantity<T, D> asin(const quantity<T, dimensionless>& q);

template <typename T, typename D>
quantity<T, D> acos(const quantity<T, dimensionless>& q);

template <typename T, typename D>
quantity<T, D> atan(const quantity<T, dimensionless>& q);

template <typename T, typename D1, typename D2>
quantity<T, D1> atan2(const quantity<T, D1>& y, const quantity<T, D2>& x);

// Hyperbolic functions
template <typename T, typename D>
quantity<T, dimensionless> sinh(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> cosh(const quantity<T, D>& q);

template <typename T, typename D>
quantity<T, dimensionless> tanh(const quantity<T, D>& q);

// Inverse hyperbolic functions
template <typename T, typename D>
quantity<T, D> asinh(const quantity<T, dimensionless>& q);

template <typename T, typename D>
quantity<T, D> acosh(const quantity<T, dimensionless>& q);

template <typename T, typename D>
quantity<T, D> atanh(const quantity<T, dimensionless>& q);
```

### Linear Algebra Operations

The SQUINT library provides various linear algebra operations for tensors:

```cpp
// Element-wise operations
template <typename T, layout L, error_checking E, size_t... Dims>
auto operator+(const fixed_tensor<T, L, E, Dims...>& lhs, const fixed_tensor<T, L, E, Dims...>& rhs);

template <typename T, layout L, error_checking E, size_t... Dims>
auto operator-(const fixed_tensor<T, L, E, Dims...>& lhs, const fixed_tensor<T, L, E, Dims...>& rhs);

template <typename T, layout L, error_checking E, size_t... Dims>
auto operator*(const fixed_tensor<T, L, E, Dims...>& lhs, const fixed_tensor<T, L, E, Dims...>& rhs);

// Matrix multiplication
template <typename T, layout L, error_checking E, size_t M, size_t N, size_t P>
auto matmul(const fixed_tensor<T, L, E, M, N>& lhs, const fixed_tensor<T, L, E, N, P>& rhs);

// Transposition
template <typename T, layout L, error_checking E, size_t... Dims>
auto transpose(const fixed_tensor<T, L, E, Dims...>& tensor);

// Matrix inversion
template <typename T, layout L, error_checking E, size_t N>
auto inv(const fixed_tensor<T, L, E, N, N>& matrix);

// Pseudo-inverse
template <typename T, layout L, error_checking E, size_t M, size_t N>
auto pinv(const fixed_tensor<T, L, E, M, N>& matrix);

// Solve linear system
template <typename T, layout L, error_checking E, size_t N>
auto solve(const fixed_tensor<T, L, E, N, N>& A, const fixed_tensor<T, L, E, N>& b);

// Solve linear least squares
template <typename T, layout L, error_checking E, size_t M, size_t N>
auto solve_lls(const fixed_tensor<T, L, E, M, N>& A, const fixed_tensor<T, L, E, M>& b);

// Cross product (for 3D vectors)
template <typename T, layout L, error_checking E>
auto cross(const fixed_tensor<T, L, E, 3>& lhs, const fixed_tensor<T, L, E, 3>& rhs);
```

These operations are also available for `dynamic_tensor` with appropriate interfaces.

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
