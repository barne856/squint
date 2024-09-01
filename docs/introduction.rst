
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

Overview
--------

SQUINT is divided into four main components:

1. **Core**: The core module provides the foundational concepts and utilities for the library, including error checking, memory management, and layout management.

2. **Quantity**: The quantity module defines the quantity system, which represents values with associated dimensions. It includes support for common arithmetic operations, unit conversions, and compile-time dimensional analysis.

3. **Tensor**: The tensor module implements the tensor system, which provides a flexible, policy-based tensor class for numerical computations. It includes support for BLAS backends, tensor operations, and views.

4. **Geometry**: The geometry module offers geometric utilities, such as projections and transformations, for use in graphics programming and physics simulations.

.. rst-class:: only-light

   .. tikz:: SQUINT Overview
        :libs: mindmap, trees, arrows
        :xscale: 80

        \begin{tikzpicture}[mindmap, grow cyclic, every node/.style=concept, concept color=blue!40,
            level 1/.append style={level distance=4.5cm,sibling angle=90},
            level 2/.append style={level distance=3cm,sibling angle=45},
            level 3/.append style={level distance=2cm,sibling angle=30}]

        \node[concept] {Squint Library}
            child[concept color=green!50] { node {Core}
                child { node {Concepts} }
                child { node {Error Checking} }
                child { node {Layout} }
                child { node {Memory} }
            }
            child[concept color=red!50] { node {Quantity}
                child { node {Dimension} }
                child { node {Unit} }
                child { node {Constants} }
                child { node {Ops} }
            }
            child[concept color=orange!50] { node {Tensor}
                child { node {Tensor Class} }
                child { node {Ops} }
                child { node {Views} }
                child { node {BLAS Backend} }
            }
            child[concept color=purple!50] { node {Geometry}
                child { node {Proj} }
                child { node {Trans} }
            };
        \end{tikzpicture}


.. rst-class:: only-dark

   .. tikz:: SQUINT Overview
      :libs: mindmap, trees, arrows
      :xscale: 80

        \begin{tikzpicture}[mindmap, grow cyclic, every node/.style=concept, concept color=blue!40,
            level 1/.append style={level distance=4.5cm,sibling angle=90},
            level 2/.append style={level distance=3cm,sibling angle=45},
            level 3/.append style={level distance=2cm,sibling angle=30}]

        \node[concept] {Squint Library}
            child[concept color=green!50] { node {Core}
                child { node {Concepts} }
                child { node {Error Checking} }
                child { node {Layout} }
                child { node {Memory} }
            }
            child[concept color=red!50] { node {Quantity}
                child { node {Dimension} }
                child { node {Unit} }
                child { node {Constants} }
                child { node {Ops} }
            }
            child[concept color=orange!50] { node {Tensor}
                child { node {Tensor Class} }
                child { node {Ops} }
                child { node {Views} }
                child { node {BLAS Backend} }
            }
            child[concept color=purple!50] { node {Geometry}
                child { node {Proj} }
                child { node {Trans} }
            };
        \end{tikzpicture}
