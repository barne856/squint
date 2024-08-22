
Building and Testing
====================



Compiler Support
----------------


SQUINT leverages modern C++ features and requires a C++23 compliant compiler. Currently supported compilers include:

- GCC (g++) version 12 or later
- Clang version 15 or later

Note: MSVC is partially supported but lacks support for multidimensional subscript operators.


Build Instructions
------------------


1. Ensure you have CMake version 3.28 or later and a supported compiler installed.
2. Optionally install MKL or OpenBLAS for BLAS backend support.
3. Build the project using the following commands:

.. code-block::

   mkdir build && cd build
   cmake ..
   cmake --build .


CMake Configuration
-------------------


SQUINT provides several CMake options to customize the build:

- `-DSQUINT_BLAS_BACKEND`: Choose the BLAS backend (MKL, OpenBLAS, or NONE)
- `-DSQUINT_BUILD_DOCUMENTATION`: Build the documentation files (ON/OFF)
- `-DSQUINT_BUILD_TESTS`: Enable/disable building tests (ON/OFF)
- `-DCMAKE_BUILD_TYPE`: Set the build type (Debug, Release, etc.)


BLAS Backends
-------------


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


Serving Documentation
---------------------


If SQUINT was built with documentation, you can serve it locally using

.. code-block::

   python3 -m http.server 80 -d ./build/sphinx

