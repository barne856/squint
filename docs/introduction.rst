
Introduction
============


SQUINT (Static Quantities in Tensors) is a header-only C++ library designed for compile-time dimensional analysis, unit conversion, and linear algebra operations. It combines a quantity system for handling physical units and dimensions with a tensor system for numerical computations.

SQUINT was developed primarily to suit my personal needs and preferences. It is not designed to be the fastest or most straightforward tensor library available. Instead, it prioritizes type safety, expressiveness, and a cohesive API that integrates well with physical simulations and graphics programming.

The primary goals of SQUINT are:

1. To provide a type-safe framework for calculations involving physical quantities, catching dimension-related errors at compile-time where possible.
2. To offer a tensor system with an API that balances ease of use with static type checking.
3. To integrate seamlessly with physical quantities, enabling tensor operations on dimensioned values.
4. To make an honest effort at *good* performance.

SQUINT is particularly suited for projects where type safety and dimensional correctness are important, such as physics engines, scientific simulations, or graphics applications dealing with real-world units. It aims to catch errors early in the development process while providing a comfortable API for both quantities and tensors.

While the library makes efforts to be performant, especially through the use of compile-time optimizations and BLAS integration, users requiring absolute peak performance or a minimalist API might find other specialized libraries more suitable for their needs.

