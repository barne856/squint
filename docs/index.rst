Welcome to SQUINT's documentation!
==================================

SQUINT (Static QUantities IN Tensors) is a C++ library for handling physical quantities with units, dimensional analysis, and linear algebra operations. It provides a type-safe and efficient way to work with physical quantities in scientific and engineering applications.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   basic_usage
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core/concepts
   api/core/error_checking
   api/core/layout
   api/core/memory
   api/quantity/quantity
   api/quantity/dimension
   api/quantity/unit
   api/quantity/constants
   api/quantity/math
   api/tensor/tensor_base
   api/tensor/fixed_tensor
   api/tensor/dynamic_tensor
   api/tensor/tensor_view
   api/tensor/tensor_ops
   api/linalg/blas_interface
   api/linalg/lapack_interface
   api/linalg/vector_ops
   api/linalg/matrix_ops
   api/geometry/transformations
   api/geometry/projections
   api/util/sequence_utils
   api/util/math_utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Features
--------

- Dimensional analysis and unit conversion
- Mathematical operations with physical quantities
- Linear algebra operations with units
- Geometry calculations with physical quantities
- Tensor operations
- BLAS and LAPACK interfaces
- Comprehensive set of physical constants
- Error checking and overflow protection
- Support for both fixed and dynamic tensors
- Flexible memory layout options

Quick Example
-------------

Here's a simple example of using SQUINT to perform calculations with physical quantities:

.. code-block:: cpp

   #include <squint/squint.hpp>
   #include <iostream>

   int main() {
       using namespace squint;
       using namespace squint::units;

       // Define some quantities
       auto distance = 100.0 * meters;
       auto time = 9.8 * seconds;

       // Calculate velocity
       auto velocity = distance / time;

       std::cout << "Velocity: " << velocity << " m/s" << std::endl;

       // Use mathematical functions
       auto angle = 45.0 * degrees;
       auto sine_value = sin(angle);

       std::cout << "Sine of 45 degrees: " << sine_value << std::endl;

       // Use physical constants
       std::cout << "Speed of light: " << si_constants<double>::c << " m/s" << std::endl;

       return 0;
   }

Installation
------------

For detailed installation instructions, see :doc:`installation`.

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details on how to get involved.

License
-------

SQUINT is distributed under the MIT License. See the LICENSE file for more information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`