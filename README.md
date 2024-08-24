# SQUINT (Static Quantities in Tensors)

SQUINT is a header-only C++ library designed for compile-time dimensional analysis, unit conversion, and linear algebra operations. It's particularly well-suited for graphics programming and physics simulations, combining a quantity system for handling physical units and dimensions with a tensor system for efficient numerical computations.

## Key Features

- Compile-time dimensional analysis
- Flexible tensor system supporting both fixed and dynamic shapes
- Integration of physical quantities with tensor operations
- Optional runtime error checking
- Support for common linear algebra operations
- Useful mathematical and physical constants

## Installation

SQUINT is a header-only library. To use it in your project:

1. Copy the `include/squint` directory to your project's include path.
2. Include the necessary headers in your C++ files:

```cpp
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>
#include <squint/geometry.hpp>
```

For CMake projects, you can use FetchContent for a more streamlined integration:

```cmake
include(FetchContent)

FetchContent_Declare(
    squint
    GIT_REPOSITORY https://github.com/barne856/squint.git
    GIT_TAG main  # or a specific tag/commit
)

FetchContent_MakeAvailable(squint)

target_link_libraries(your_target PRIVATE squint::squint)
```

## Examples

SQUINT can be used for common graphics operations:

```cpp
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>
#include <squint/geometry.hpp>
#include <iostream>

using namespace squint;

int main() {
    // Define a 3D point
    vec3_t<length> point{
        length(1.0f),
        length(2.0f),
        length(3.0f)
    };

    // Create a model matrix
    mat4 model = mat4::eye();

    // Apply transformations
    geometry::translate(
        model,
        vec3{
            length(2.0f),
            length(1.0f),
            length(0.0f)
        }
    );
    geometry::rotate(
        model,
        math_constants::pi<float> / 4.0f,
        vec3{0.0f, 1.0f, 0.0f}
    );
    geometry::scale(
        model,
        vec3{2.0f, 2.0f, 2.0f}
    );

    // Transform the point
    vec4 homogeneous_point{point(0), point(1), point(2), 1.0f};
    vec4 transformed_point = model * homogeneous_point;

    return 0;
}
```

## Physics Simulation Example

SQUINT can be used for physics calculations:

```cpp
#include <squint/quantity.hpp>
#include <squint/tensor.hpp>
#include <iostream>

using namespace squint;

int main() {
    // Define initial conditions
    auto pos = vec3_t<length>::zeros();
    vec3_t<velocity> vel{
        velocity(5.0f),
        velocity(10.0f),
        velocity(0.0f)
    };
    vec3_t<acceleration_t<float>> acc{
        acceleration(0.0f),
        acceleration(-9.81f),
        acceleration(0.0f)
    };

    // Simulation parameters
    auto dt = squint::time::seconds(0.1f);
    auto total_time = squint::time::seconds(1.0f);

    // Simulation loop
    for (auto t = squint::time::seconds(0.0f); t < total_time; t += dt) {
        // Update position and velocity
        position += velocity * dt + 0.5f * acceleration * dt * dt;
        velocity += acceleration * dt;

        // Print current state
        std::cout << "Time: " << t << "\nPosition: " << position << "\nVelocity: " << velocity << "\n\n";
    }

    return 0;
}
```

## Documentation

For detailed information about SQUINT's features, API reference, and advanced usage, please refer to the [full documentation](https://barne856.github.io/squint/).

## License

MIT