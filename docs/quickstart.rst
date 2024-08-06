Quickstart
==========

This quickstart guide will help you get up and running with SQUINT.

Basic Usage
-----------

1. Include the necessary headers:

   .. code-block:: cpp

      #include <squint/squint.hpp>

2. Use the `squint` namespace:

   .. code-block:: cpp

      using namespace squint;
      using namespace squint::units;

3. Create quantities:

   .. code-block:: cpp

      auto distance = 100.0 * meters;
      auto time = 10.0 * seconds;

4. Perform calculations:

   .. code-block:: cpp

      auto velocity = distance / time;

5. Use mathematical functions:

   .. code-block:: cpp

      auto angle = 45.0 * degrees;
      auto sine_value = sin(angle);

6. Use physical constants:

   .. code-block:: cpp

      auto speed_of_light = si_constants<double>::c;

Example
-------

Here's a complete example:

.. code-block:: cpp

   #include <squint/squint.hpp>
   #include <iostream>

   int main() {
       using namespace squint;
       using namespace squint::units;

       auto distance = 100.0 * meters;
       auto time = 10.0 * seconds;
       auto velocity = distance / time;

       std::cout << "Velocity: " << velocity << " m/s" << std::endl;

       auto angle = 45.0 * degrees;
       auto sine_value = sin(angle);

       std::cout << "Sine of 45 degrees: " << sine_value << std::endl;

       std::cout << "Speed of light: " << si_constants<double>::c << " m/s" << std::endl;

       return 0;
   }