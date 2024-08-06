Basic Usage
===========

Working with Quantities
-----------------------

Creating Quantities
^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   auto length = 5.0 * meters;
   auto mass = 10.0 * kilograms;
   auto time = 2.0 * seconds;

Basic Arithmetic
^^^^^^^^^^^^^^^^

.. code-block:: cpp

   auto area = length * length;
   auto volume = length * length * length;
   auto speed = length / time;
   auto acceleration = speed / time;
   auto force = mass * acceleration;

Unit Conversions
----------------

.. code-block:: cpp

   auto distance_m = 1000.0 * meters;
   auto distance_km = distance_m / kilometers;
   auto distance_mi = distance_m / miles;

Mathematical Functions
----------------------

.. code-block:: cpp

   auto angle = 45.0 * degrees;
   auto sine_value = sin(angle);
   auto cosine_value = cos(angle);

Using Constants
---------------

.. code-block:: cpp

   auto speed_of_light = si_constants<double>::c;
   auto planck_constant = si_constants<double>::h;
   auto gravitational_constant = si_constants<double>::G;

Error Checking
--------------

.. code-block:: cpp

   try {
       auto result = checked_quantity_t<int, dimensions::L>(std::numeric_limits<int>::max()) + 1 * meters;
   } catch (const std::overflow_error& e) {
       std::cerr << "Overflow detected: " << e.what() << std::endl;
   }