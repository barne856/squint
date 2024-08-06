Installation
============

Prerequisites
-------------

To use SQUINT, you need:

- A C++20 compatible compiler
- CMake (version 3.28 or higher)

Building from Source
--------------------

1. Clone the repository:
   
   .. code-block:: bash

      git clone https://github.com/your-username/squint.git
      cd squint

2. Create a build directory:

   .. code-block:: bash

      mkdir build
      cd build

3. Configure the project:

   .. code-block:: bash

      cmake ..

4. Build the library:

   .. code-block:: bash

      cmake --build .

5. (Optional) Install the library:

   .. code-block:: bash

      sudo cmake --install .

Using SQUINT in Your Project
----------------------------

To use SQUINT in your CMake project, add the following to your CMakeLists.txt:

.. code-block:: cmake

   find_package(SQUINT REQUIRED)
   target_link_libraries(your_target PRIVATE SQUINT::SQUINT)