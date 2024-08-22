
Installation
============


SQUINT is a header-only library, making it easy to integrate into your projects. To use it:

1. Copy the `include/squint` directory to your project's include path.
2. Include the necessary headers in your C++ files:

.. code-block::

   #include <squint/quantity.hpp>
   #include <squint/tensor.hpp>

For CMake projects, you can use FetchContent for a more streamlined integration:

.. code-block::

   include(FetchContent)
   
   FetchContent_Declare(
       squint
       GIT_REPOSITORY https://github.com/barne856/squint.git
       GIT_TAG main  # or a specific tag/commit
   )
   
   FetchContent_MakeAvailable(squint)
   
   target_link_libraries(your_target PRIVATE squint::squint)

