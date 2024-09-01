GPU Tensor Usage
================

SQUINT provides support for GPU computations using CUDA. This feature allows you to create and manipulate tensors on the GPU, enabling faster calculations for certain types of operations.

Enabling GPU Support
--------------------

To use GPU tensors, you must build SQUINT with CUDA support enabled. Use the following CMake option when building the library:

.. code-block:: bash

   cmake -DSQUINT_USE_CUDA=ON ..

This option requires that you have CUDA installed on your system. On ubnuntu, you can install CUDA using the following command:

.. code-block:: bash

   sudo apt install nvidia-cuda-toolkit

Creating GPU Tensors
--------------------

GPU tensors cannot be created directly. Instead, you create a host tensor and then transfer it to the device using the ``to_device()`` method:

.. code-block:: cpp

   // Create a host tensor
   tensor<float, shape<3, 3>> host_tensor = {1, 2, 3, 4, 5, 6, 7, 8, 9};

   // Transfer to GPU
   auto gpu_tensor = host_tensor.to_device();

Transferring Back to Host
-------------------------

To bring a GPU tensor back to the host, use the ``to_host()`` method:

.. code-block:: cpp

   auto result_host = gpu_tensor.to_host();

Supported Operations
--------------------

GPU tensors support most operations available for host tensors, with some limitations:

Supported:
^^^^^^^^^^

- Element-wise operations
- Scalar operations
- Matrix multiplication
- Reshaping
- Creating subviews

.. code-block:: cpp

   // Element-wise addition
   auto sum = gpu_tensor1 + gpu_tensor2;

   // Scalar multiplication
   auto scaled = gpu_tensor * 2.0f;

   // Matrix multiplication
   auto product = gpu_tensor1 * gpu_tensor2;

   // Reshaping
   auto reshaped = gpu_tensor.reshape<9>();

   // Creating a subview
   auto subview = gpu_tensor.subview<shape<2, 2>>({0, 0});

Not Supported:
^^^^^^^^^^^^^^

- Subscript operators (``[]`` and ``()``)
- Iteration over elements, rows, columns, or subviews
- Tensor math functions (e.g., ``solve``, ``inv``, etc.)

Performance Considerations
--------------------------

GPU tensors can significantly accelerate certain operations, especially for large datasets. However, the overhead of transferring data between the host and device should be considered. For small tensors or infrequent operations, the transfer time might outweigh the computational benefits.

Example: Matrix Multiplication on GPU
-------------------------------------

Here's a complete example demonstrating matrix multiplication on the GPU:

.. code-block:: cpp

   #include <squint/tensor.hpp>

   int main() {
       // Create host tensors
       tensor<float, shape<3, 3>> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
       tensor<float, shape<3, 3>> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};

       // Transfer to GPU
       auto A_gpu = A.to_device();
       auto B_gpu = B.to_device();

       // Perform matrix multiplication on GPU
       auto C_gpu = A_gpu * B_gpu;

       // Transfer result back to host
       auto C = C_gpu.to_host();

       // Print result
       std::cout << "Result of matrix multiplication:" << std::endl;
       std::cout << C << std::endl;

       return 0;
   }

Best Practices
--------------

1. Minimize data transfers between host and device to reduce overhead.
2. Use GPU tensors for computationally intensive operations on large datasets.
3. Batch operations when possible to maximize GPU utilization.
4. Profile your code to ensure that GPU operations are providing a performance benefit.
