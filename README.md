# SQUINT - Static Quantities in Tensors

SQUINT is a header only C++ library for compile-time physical quantities and tensors. The library also contains some helpful linear algebra, numerical integration, and root finding / optimization algorithms (including autodiff and numerical differentiation) using these abstractions.

SQUINT allows for strict compile-time checking of the dimensions of phyical quantities such that formulas you implement make physical sense. These quantities can also be used as elements of tensors so that formulas involving vectors and matricies can also be statically type checked. The library's main purpose is for use in the implementation of physical simulations and 3D rendering where keeping track of physical dimensions can be important and algorithms can be more elegantly implemented using concepts from linear algebra.

Below is example code of a statically type checked formula involving tensors of physical quantites.

```cpp
#include "squint/linalg.hpp"
using namespace squint;
using namespace squint::quantities;
int main()
{
    using time = squint::quantities::time;
    // time matrix
    tensor<time, 3, 3> A{
        tensor<time, 1, 3>{1_s, 5_s, 6_s},
        tensor<time, 1, 3>{8_s, 2_s, 3_s},
        tensor<time, 1, 3>{3_s, 5_s, 9_s}};
    // length column vector
    tensor<length, 3> b{1_m, 2_m, 3_m};
    // velocity column vector
    auto x = b / A; // operator/ solves the general linear least squares problem
    // print the solution in feet per second
    for (const velocity &vi : x)
        std::cout << vi.as_fps() << std::endl;
    return 0;
}
```

## Dependencies

- C++20 compiler
- Intel MKL (optional)

## How to Build

SQUINT has no required external dependencies other than the C++20 standard library. Simply include the header files in your project if you only need the physical quantities and tensor data structures. The library can optionally be linked with Intel's Math Kernel Libray (MKL) to speed up some operations. It is highly recommended to link with MKL if you will use the library to perform large linear algrebra computations because little care has been taken to make the non-MKL implementaiton of these algorithms efficient.

An example CMakeLists.txt is included that builds the tests and links with MKL if you set the CMake flag `USE_MKL`.

## Constructors

Tensors are stored internally in column major order. You can create a matrix by supplying an initalizer list of elements in column major order.

```cpp
tensor<double, 3, 3> A{1,2,3,4,5,6,7,8,9};
```

Alternativly, you can provide an initalizer list of equal sized tensors as long as the shapes and amounts exactally fit the shape of the new tensor. The elements are assigned as blocks again in column major order. For example, to construct a 3x3 matrix from 3 1x3 row vectors:

```cpp
tensor<double, 3, 3> B{
    tensor<double, 1, 3>{1,4,7},
    tensor<double, 1, 3>{2,5,8},
    tensor<double, 1, 3>{3,6,9},
};
// A == B will be true
```

Or, to construct the same matrix again using column vectors:

```cpp
tensor<double, 3, 3> C{
    tensor<double, 3>{1,2,3},
    tensor<double, 3>{4,5,6},
    tensor<double, 3>{7,8,9},
};
// B == C will be true
```

## Indexing

Indexing tensors works similarly to multidimensional arrays:

```cpp
tensor<double, 1, 3> row0 = A[0]; // get zeroth row of A
tensor<double> elem01 = row0[1]; // get first element of row
```

You can also index by block:

```cpp
// get a 2x2 block offset by 1,1
tensor<double, 2, 2> block = A.at<2,2>(1,1);
```

You can also iterate over all the elements of a tensor:

```cpp
for(auto& elem : A)
{
    elem*=2;
}
```

Or, iterate over sub-matrices or blocks of a tensor:

```cpp
for(const auto& row : A.block<1,3>())
{
    std::cout << row << std::endl;
}
```

For convience, there are cols() and rows() methods that can be used for tensors of order <= 2:

```cpp
for(const auto& row : A.rows())
{
    std::cout << row << std::endl;
}
```

## Operators

Most of the math operators are in the `linalg.hpp` header. When you use dimensionful quantities as elements to tensors, the operators will deduce the resultant types.

```cpp
quantity<double, time> t{1};
quantity<double, length> x{1};
quantity<double, velocity> v = x/t; // length/time -> velocity
auto A = tensor<quantity<double, length>, 4,4>::I(); // identity matrix
auto A_inv = inv(A); // A_inv has dimension length^(-1)
```

## Dynamic Shape Tensors

All operations for tensors have corresponding run-time equivalents. Typically the API for these methods is the same only the compile-time arguments in the `<>` brackets are instead run time arguments to the function calls. This allows for tensors that are too large to fit on the stack or need to change shape at run-time.

Dynamic Shape Tensors are created and reshaped like so:

```cpp
// create a 4x4 matrix filled with ones.
tensor<double, dynamic_shape> ones({4, 4}, 1);
A.reshape({8,2}); // reshape to 8x2 matrix
```

## Accessing Data
A pointer to the underlying data of a tensor can be accessed through the `data()` method. For fixed size and dynamic sized tensors, the data is guaranteed to be contiguous and stored in column major order. If the underlying data type of the tensor is a `quantity`, a pointer to a `quantity` is returned. `quantity` pointers can be explicitly converted their underlying type (usually float or double) using a `static_cast` or C style cast.

A tensor of fixed size tensors is also guaranteed to have all its `data()` stored contiguously with no overhead of virtual function tables or other member variables and the size of a fixed size tensor is equal to the size of its elements.

## Tensor Refs
`tensor_ref`s are returned from some operations on tensors such as indexing or transposing. These reference tensors which store the raw data, but at different strides and offsets from the main tensor. The `data()` of `tensor_ref`s are not necessarily contiguous or stored in column major order. You must copy the results to a new `tensor` by using the `copy()` method or by using a constructor for a `tensor` to have this guarantee.

## Performance Limitations
Performance of linear algebra operations are similar to what you would expect from BLAS and LAPACKE for single unfused operations. Note that indexing dynamic shape tensors and tensor_refs with the `operator[]` is quite slow since the location of the elements must be computed at run-time. It is best to not index dynamic shape tensors in hot parts of your code. For similar reasons, iterators of tensors can be slower than iterators over the flat array.