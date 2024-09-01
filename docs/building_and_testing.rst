Building and Testing
====================

Compiler Support
----------------

SQUINT leverages modern C++ features and requires a C++23 compliant compiler. Currently supported compilers include:

- GCC (g++) version 12 or later
- Clang version 15 or later

.. note::
   MSVC is partially supported but lacks support for multidimensional subscript operators.

Build Instructions
------------------

1. Ensure you have CMake version 3.28 or later and a supported compiler installed.
2. Optionally install MKL if you intend to use it as a BLAS backend.
3. If you want to use GPU support, ensure you have CUDA installed on your system.
4. Build the project using the following commands:

.. code-block:: bash

   mkdir build && cd build
   cmake ..
   cmake --build .

CMake Configuration
-------------------

SQUINT provides several CMake options to customize the build:

- ``-DSQUINT_BLAS_BACKEND``: Choose the BLAS backend (MKL, OpenBLAS, or NONE)
- ``-DSQUINT_BUILD_DOCUMENTATION``: Build the documentation files (ON/OFF)
- ``-DSQUINT_BUILD_TESTS``: Enable/disable building tests (ON/OFF)
- ``-DCMAKE_BUILD_TYPE``: Set the build type (Debug, Release, etc.)
- ``-DSQUINT_USE_CUDA``: Enable/disable CUDA support for GPU tensors (ON/OFF)

BLAS Backends
-------------

SQUINT supports three BLAS backends to cater to different performance needs and system configurations:

1. Intel MKL: Optimized for high performance on Intel processors

   .. code-block:: bash

      cmake -DSQUINT_BLAS_BACKEND=MKL ..

2. OpenBLAS: An open-source alternative that's portable across different architectures

   .. code-block:: bash

      cmake -DSQUINT_BLAS_BACKEND=OpenBLAS ..

3. NONE: A limited fallback implementation for maximum portability

   .. code-block:: bash

      cmake -DSQUINT_BLAS_BACKEND=NONE ..

.. note::
   For the OpenBLAS backend, SQUINT will automatically fetch the source code from GitHub and build it from source along with the library if you use the provided CMakeLists.txt file.

Enabling CUDA Support
---------------------

To enable CUDA support for GPU tensors, use the following CMake option:

.. code-block:: bash

   cmake -DSQUINT_USE_CUDA=ON ..

Ensure that you have CUDA installed on your system before enabling this option.

Serving Documentation
---------------------

If SQUINT was built with documentation, you can serve it locally using:

.. code-block:: bash

   python -m http.server -d ./build/sphinx