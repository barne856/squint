Advanced Usage
==============

Custom Dimensions and Units
---------------------------

Creating Custom Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   using custom_dimension = dimension<std::ratio<1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

Creating Custom Units
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   template <typename T>
   using custom_unit = unit<T, custom_dimension, 2.5>; // 2.5 is the scale factor

Working with Tensors
--------------------

Fixed Tensors
^^^^^^^^^^^^^

.. code-block:: cpp

   // Example to be added when fixed_tensor implementation is available

Dynamic Tensors
^^^^^^^^^^^^^^^

.. code-block:: cpp

   // Example to be added when dynamic_tensor implementation is available

Linear Algebra Operations
-------------------------

.. code-block:: cpp

   // Examples to be added when linear algebra operations are implemented

Geometry Calculations
---------------------

.. code-block:: cpp

   // Examples to be added when geometry calculations are implemented

Advanced Error Handling
-----------------------

.. code-block:: cpp

   // Examples of advanced error handling techniques