# SQUINT C++ Library

## Table of Contents

1. [Key Features](#key-features)
2. [Core Components](#core-components)
   - [Dimension System](#dimension-system)
   - [Quantity System](#quantity-system)
   - [Tensor System](#tensor-system)
   - [Units and Constants](#units-and-constants)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Basic Quantity Operations](#basic-quantity-operations)
   - [Using Physical Constants](#using-physical-constants)
   - [Basic Tensor Operations](#basic-tensor-operations)
   - [Linear Algebra Operations](#linear-algebra-operations)
5. [Advanced Usage](#advanced-usage)
   - [Working with Mixed Types](#working-with-mixed-types)
6. [API Reference](#api-reference)
   - [Dimension System](#dimension-system-1)
   - [Quantity System](#quantity-system-1)
   - [Tensor System](#tensor-system-1)
   - [Units Namespace](#units-namespace)
   - [Constants Namespace](#constants-namespace)
   - [Math Namespace](#math-namespace)
   - [Linear Algebra Operations](#linear-algebra-operations-1)
7. [Building and Testing](#building-and-testing)
   - [On Windows](#on-windows)
   - [On Linux](#on-linux)
   - [Running Tests](#running-tests)

## What is This?

SQUINT (Static Quantities in Tensors) is a modern, header-only C++ library designed to bring together compile-time dimensional analysis, unit conversion, and linear algebra operations in C++. Squint provides a powerful API that can enhance code safety, readability, and expressiveness without compromising performance.

With SQUINT, you can:

- Catch unit-related errors at compile-time
- Perform intuitive arithmetic with physical quantities
- Manipulate multi-dimensional data
- Conduct complex linear algebra operations safely and efficiently

Whether you're simulating physical systems, rendering 3D geometry, or developing cutting-edge algorithms, SQUINT provides the tools you need to write cleaner, safer, and more expressive code.

## Key Features

- **Dimensional Analysis**: Catch unit-related errors before they become runtime bugs.
- **Unit Conversions**: Say goodbye to manual unit conversion headaches.
- **Quantity System**: Represent physical quantities with built-in unit information.
- **Tensor Operations**: Handle multi-dimensional data with both fixed-size and dynamic tensors (n-d arrays).
- **Error Checking**: Optional runtime checks for operations like overflow and division by zero.
- **Constants and Functions**: A comprehensive set of mathematical and physical constants.
- **Performance-Oriented**: Designed to minimize runtime overhead without sacrificing functionality.

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
    auto subview = t.subview<2, 2>(0, 1);

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

### Tensor Views and Reshaping

SQUINT provides powerful capabilities for creating views and subviews of tensors, as well as reshaping operations. These features allow for efficient and flexible manipulation of tensor data without copying the underlying memory.

#### Creating Views

```cpp
mat3x4_t<float> tensor{{/* ... */}};

// Create a view of the entire tensor
auto view = tensor.view();
```

A view provides a non-owning reference to the tensor data, allowing you to perform operations on the tensor without modifying its shape or storage.

#### Subviews

Subviews allow you to work with a portion of a tensor. They are created using the `subview` method, which takes subview shape and offsets as arguments:

```cpp
// Create a subview from a dynamic shape tensro
auto dynamic_subview = tensor.subview({2,2}, {1,2});
// Fixed shape tensor (only offsets are required since the shape is a compile time constant)
auto subview = tensor.subview<2, 2>(1, 2);
```

The template arguments to `subview<2, 2>` specify the dimensions of the resulting subview.

Here's a more detailed example:

```cpp
// column major order
mat4_t<float> matrix{{
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
}};
// matrix contains:
// [
//    [1, 5, 9, 13],
//    [2, 6, 10, 14],
//    [3, 7, 11, 15],
//    [4, 8, 12, 16]
// ]

// Create a 2x3 subview starting from row 1, column 1
auto subview = matrix.subview<2, 3>(1, 1);

// subview now contains:
// [
//    [6, 10, 14],
//    [7, 11, 15]
// ]
```

#### Subview Iterations

SQUINT provides convenient methods to iterate over subviews of a tensor:

```cpp
// Iterate over all 2x2 subviews of the matrix
for (auto subview : matrix.subviews<2, 2>()) {
    // Process each 2x2 subview
}

// Iterate over rows
for (auto row : matrix.rows()) {
    // Process each row
}

// Iterate over columns
for (auto col : matrix.cols()) {
    // Process each column
}

// Get a specific row or column
auto thirdRow = matrix.row(2);
auto secondColumn = matrix.col(1);
```

#### Reshaping

Reshaping allows you to change the dimensions of a tensor without changing its data:

```cpp
mat3x4_t<float> tensor{{/* ... */}};

// Reshape the 3x4 tensor into a 6x2 tensor
auto reshaped = tensor.reshape<6, 2>();

// Flatten the tensor into a 1D vector
auto flattened = tensor.flatten();
```

When reshaping, the total number of elements must remain the same. The `reshape` method returns a view of the original data with the new shape.

#### Combining Operations

These operations can be combined for powerful data manipulation:

```cpp
mat4_t<float> bigMatrix{{/* ... */}};

// Create a view of the upper-left 3x3 submatrix, then flatten it
auto flattened_subview = bigMatrix.subview<3, 3>(0, 0).flatten();

// Reshape a subview
auto reshaped_subview = bigMatrix.subview<2, 4>(1, 0).reshape<4, 2>();
```

These advanced features allow for efficient and expressive tensor manipulations, enabling complex operations without unnecessary data copying.


### Tensor Division Operator

SQUINT implements a powerful and flexible `operator/` for tensors, which performs general matrix division. This operation is equivalent to solving a general linear system or finding the minimum norm solution, depending on the shapes of the tensors involved.

#### Functionality

The `operator/` for tensors in SQUINT is designed to handle various scenarios:

1. **General Linear System**: When the dimensions are compatible for a standard linear system (Ax = B).
2. **Overdetermined System**: When there are more equations than unknowns, resulting in a least squares solution.
3. **Underdetermined System**: When there are fewer equations than unknowns, resulting in a minimum norm solution.

#### Usage

The general syntax for using the division operator is:

```cpp
auto x = B / A;
```

Where `A` and `B` are tensors, and the operation solves the system Ax = B for x.

#### Behavior Based on Tensor Shapes

1. **Square Matrices (N x N)**:
   - Performs standard matrix division (equivalent to A⁻¹B).
   - Example:
     ```cpp
     mat3 A{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
     vec3 B{{1, 2, 3}};
     auto x = B / A;  // Solves Ax = B
     ```

2. **Overdetermined System (M x N, where M > N)**:
   - Computes the least squares solution.
   - Minimizes ||Ax - B||².
   - Example:
     ```cpp
     mat<4, 3> A{{/* ... */}};
     vec<4> B{{/* ... */}};
     auto x = B / A;  // Least squares solution
     ```

3. **Underdetermined System (M x N, where M < N)**:
   - Computes the minimum norm solution.
   - Finds x with minimum ||x|| that satisfies Ax = B.
   - Example:
     ```cpp
     mat<3, 4> A{{/* ... */}};
     vec<3> B{{/* ... */}};
     auto x = B / A;  // Minimum norm solution
     ```

4. **Matrix-Matrix Division**:
   - Solves AX = B for X, where A, B, and X are matrices.
   - Example:
     ```cpp
     mat3 A{{/* ... */}};
     mat<3, 2> B{{/* ... */}};
     auto X = B / A;  // Solves AX = B
     ```

#### Implementation Details

- The `operator/` uses efficient LAPACK routines under the hood for numerical stability and performance.
- For overdetermined systems, it uses a QR factorization with column pivoting.
- For underdetermined systems, it uses an LQ factorization.

#### Error Handling

- If the system is singular (i.e., A is not invertible for square matrices), a `std::runtime_error` is thrown.
- For error checked types, the operation checks for compatible dimensions and throws a `std::invalid_argument` if the shapes are incompatible.

#### Performance Considerations

- The division operator is more computationally expensive than basic arithmetic operations.
- For repeated solutions with the same left-hand side (A), consider using SQUINT's `solve` or `solve_lls` functions directly for better performance.

This tensor division operator provides a intuitive and mathematically consistent way to solve linear systems and perform matrix divisions in SQUINT, handling various cases seamlessly based on the shapes of the input tensors.

#### Dimensional Consistency with Quantities

One of the powerful features of SQUINT's tensor division operator is its ability to maintain dimensional consistency when working with tensors of quantities. This ensures that the results of your calculations are not only mathematically correct but also physically meaningful.

- **Automatic Dimension Handling**: When you use `operator/` with tensors of quantities, SQUINT automatically handles the dimensions, ensuring that the resulting tensor has the correct units.

- **Error Prevention**: The dimensional analysis helps prevent errors in physical calculations by catching dimensionally inconsistent operations at compile-time.

- **Intuitive Results**: The resulting dimensions follow the expected physical laws, making the code more intuitive and self-documenting.

Example of dimensionally consistent division:

```cpp
// Define a 3x3 matrix of quantities with units of length/time (e.g., velocity)
mat3_t<velocity> A{{
    velocity::meters_per_second(1), velocity::meters_per_second(2), velocity::meters_per_second(3),
    velocity::meters_per_second(4), velocity::meters_per_second(5), velocity::meters_per_second(6),
    velocity::meters_per_second(7), velocity::meters_per_second(8), velocity::meters_per_second(9)
}};

// Define a vector of quantities with units of length
vec3_t<length> B{{
    length::meters(1),
    length::meters(2),
    length::meters(3)
}};

// Perform the division
auto X = B / A;

// X now contains quantities with units of time
// This is dimensionally consistent: length / (length/time) = time
```

In this example:
- `A` has units of velocity (length/time)
- `B` has units of length
- The resulting `X` has units of time, which is dimensionally consistent with solving the equation Ax = B

This dimensional consistency applies to all cases of the tensor division operator, including:
- Square matrix division
- Overdetermined systems (least squares solutions)
- Underdetermined systems (minimum norm solutions)
- Matrix-matrix divisions

#### Compile-Time Dimension Checking

SQUINT performs dimension checking at compile-time, which means:
- Dimensionally incorrect operations will result in compile-time errors, catching potential mistakes early in the development process.
- There's no runtime overhead for dimension checking, maintaining zero overhead.

#### Note on Dimensionless Quantities

When working with dimensionless quantities or mixing dimensioned and dimensionless quantities, SQUINT handles these cases appropriately. Dimensionless quantities can be used freely in any position without affecting the dimensional analysis and built in arithmetic types are treated as dimensionless.

### Working with Mixed Types

Squint supports operations between quantities with different underlying types:

```cpp
quantity<double, length> l_double(5.0);
quantity<float, length> l_float(3.0F);
quantity<int, length> l_int(2);

auto result = l_double + l_float + l_int; // result is quantity<double, length>
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
T& at(size_t... indices);  // Element access
T& operator[](size_t... indices);  // Element access
T* data() noexcept;  // Returns pointer to underlying data
auto subview(Offsets... start);  // Creates subview of tensor
template<size_t... SubviewShape>
auto subviews(); // Iterator of subviews
auto view();  // Creates view of entire tensor
template <size_t... NewDims>
auto reshape();  // Reshapes tensor
auto flatten();  // Returns flattened view of tensor
auto rows();  // Iterator of row views
auto cols();  // Iterator of column views
auto row(size_t index) // View of a single row
auto col(size_t index) // View of a single col
```

##### Static Methods

```cpp
static auto zeros();  // Creates tensor filled with zeros
static auto ones();  // Creates tensor filled with ones
static auto full(const T& value);  // Creates tensor filled with specific value
static auto random(T min, T max);  // Creates tensor with random values
static auto eye();  // Creates identity tensor
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


Element-wise operations:

```cpp
// Addition
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto operator+(const A &a, const B &b);

// Subtraction
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto operator-(const A &a, const B &b);

// Scalar multiplication
template <fixed_shape_tensor A, scalar Scalar>
auto operator*(const A &a, const Scalar &s);
template <fixed_shape_tensor A, scalar Scalar>
auto operator*(const Scalar &s, const A &a);

// Scalar division
template <fixed_shape_tensor A, scalar Scalar>
auto operator/(const A &a, const Scalar &s);
```

Matrix operations

```cpp
// Matrix multiplication
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto operator*(const A &a, const B &b);

// Matrix division (general linear least squares or least norm solution)
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto operator*(const A &a, const B &b);

// Solve linear system
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto solve(A &a, B &b);

// Solve linear least squares
template <fixed_tensor A, fixed_tensor B>
auto solve_lls(A& a, B& b);

// Cross product (for 3D vectors)
template <fixed_tensor A, fixed_tensor B>
auto cross(const A& lhs, const B& rhs);
```

Class Methods:

```cpp
// Transposition (returning a tensor view)
auto transpose();
// Matrix inversion (returning a new tensor)
auto inv();
// Pseudo-inverse (returning a new tensor)
auto pinv();

auto squared_norm();
auto trace();
auto mean();
auto sum();
auto min();
auto max();

template <tensor Other>
auto &operator+=(const Other &other);
template <tensor Other>
auto &operator-=(const Other &other);
template <scalar Scalar>
auto &operator*=(const Scalar &s);
template <scalar Scalar>
auto &operator/=(const Scalar &s);
```

These operations are also available for `dynamic_tensor` with appropriate interfaces and also work with tensor views.

### `solve_lls` (Linear Least Squares Solver)

#### Purpose
Solves the linear least squares problem Ax = b or finds the minimum norm solution.

#### Function Signature
```cpp
template <fixed_shape_tensor A, fixed_shape_tensor B>
void solve_lls(A &a, B &b)

template <dynamic_shape_tensor A, dynamic_shape_tensor B>
void solve_lls(A &a, B &b)
```

#### Parameters
- `a`: Input matrix A (m x n)
- `b`: Input/output matrix or vector B (m x nrhs) or (m)

#### Behavior
1. **In-place Operation**: This function overwrites both input matrices `a` and `b`.
   - Matrix `a` is overwritten with details of its factorization.
   - Matrix `b` is overwritten with the solution `x`.

2. **Size Requirements**:
   - For `b`, the user must provide a tensor large enough to accommodate the largest possible solution.
   - If m >= n (overdetermined or square system):
     - `b` should have at least n rows.
   - If m < n (underdetermined system):
     - `b` should have at least m rows.

3. **Output**:
   - The solution `x` is stored in the first n rows of `b` if m >= n, or in the first m rows of `b` if m < n.

#### Notes
- The function uses the LAPACKE_sgels/LAPACKE_dgels routines from LAPACK.
- It's the user's responsibility to ensure `b` is large enough for any case.
- The original content of `b` beyond the actual solution size is preserved.

### `solve` (Linear System Solver)

#### Purpose
Solves the linear system of equations Ax = b.

#### Function Signature
```cpp
template <fixed_shape_tensor A, fixed_shape_tensor B>
auto solve(A &a, B &b)

template <dynamic_shape_tensor A, dynamic_shape_tensor B>
auto solve(A &a, B &b)
```

#### Parameters
- `a`: Input square matrix A (n x n)
- `b`: Input/output matrix or vector B (n x nrhs) or (n)

#### Behavior
1. **In-place Operation**: This function overwrites both input matrices `a` and `b`.
   - Matrix `a` is overwritten with the factors L and U from the factorization A = P*L*U.
   - Matrix `b` is overwritten with the solution `x`.

2. **Size Requirements**:
   - Matrix `a` must be square (n x n).
   - Matrix `b` must have n rows and can have any number of columns (nrhs).

3. **Output**:
   - The solution `x` replaces `b`.
   - The function returns a vector of pivot indices.

#### Notes
- The function uses the LAPACKE_sgesv/LAPACKE_dgesv routines from LAPACK.
- `A` must be dimensionless (units cancel out) since `b` is modified in-place to become `x`.
- The pivot indices can be used to determine the permutation matrix P.

#### Error Handling
Both functions will throw a `std::runtime_error` if:
- The shapes of `a` and `b` are incompatible.
- The LAPACK routine fails (non-zero info returned).

Users should ensure proper error handling when using these functions, as they can throw exceptions.

## Building and Testing

### Compiler Support

SQUINT requires a C++ compiler with support for C++23 features, particularly multidimensional subscript operators. Currently, the library supports the following compilers:

- GCC (g++) version 12 or later
- Clang version 15 or later

#### Unsupported Compilers

Microsoft Visual C++ (MSVC) is currently **not supported** by SQUINT. This is due to MSVC's lack of support for multidimensional subscript operators, which are a key feature used in SQUINT for tensor indexing and manipulation.

#### Compiler-Specific Notes

##### GCC (g++)
- Minimum version: 12
- Fully supports all SQUINT features

##### Clang
- Minimum version: 15
- Fully supports all SQUINT features

##### MSVC
- Not currently supported
- SQUINT relies heavily on multidimensional subscript operators, which are not yet implemented in MSVC
- Support may be added in the future if MSVC implements this standard C++23 feature

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
