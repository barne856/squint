Geometry Functions in SQUINT
============================

SQUINT provides a set of geometry functions to facilitate common operations in computer graphics and 3D mathematics. These functions are primarily focused on creating projection matrices and applying transformations to 3D objects.

Projection Functions
--------------------

SQUINT offers two main projection functions: orthographic and perspective projections.

Orthographic Projection
^^^^^^^^^^^^^^^^^^^^^^^

The `ortho` function creates an orthographic projection matrix:

.. code-block:: cpp

    template <typename T>
    auto ortho(length_t<T> left, length_t<T> right, length_t<T> bottom, length_t<T> top, 
               length_t<T> near_plane, length_t<T> far_plane, 
               length_t<T> unit_length = length_t<T>{1});

This function generates a 4x4 orthographic projection matrix that maps the specified viewing frustum onto a unit cube centered at the origin.

Parameters:

- `left`, `right`: The left and right clipping plane coordinates
- `bottom`, `top`: The bottom and top clipping plane coordinates
- `near_plane`, `far_plane`: The distances to the near and far clipping planes
- `unit_length`: The unit length for the projection space (default is 1)

Example usage:

.. code-block:: cpp

    auto left = units::meters(-10.0);
    auto right = units::meters(10.0);
    auto bottom = units::meters(-10.0);
    auto top = units::meters(10.0);
    auto near = units::meters(0.1);
    auto far = units::meters(100.0);

    auto ortho_matrix = geometry::ortho(left, right, bottom, top, near, far);

Perspective Projection
^^^^^^^^^^^^^^^^^^^^^^

The `perspective` function creates a perspective projection matrix:

.. code-block:: cpp

    template <dimensionless_scalar T, typename U>
    auto perspective(T fovy, T aspect, length_t<U> near_plane, length_t<U> far_plane,
                     length_t<U> unit_length = length_t<U>{1});

This function generates a 4x4 perspective projection matrix based on the specified field of view, aspect ratio, and near and far clipping planes.

Parameters:

- `fovy`: The vertical field of view in radians
- `aspect`: The aspect ratio (width / height) of the viewport
- `near_plane`, `far_plane`: The distances to the near and far clipping planes
- `unit_length`: The unit length for the projection space (default is 1)

Example usage:

.. code-block:: cpp

    float fov = pi<float> / 4.0f; // 45 degrees
    float aspect_ratio = 16.0f / 9.0f;
    auto near = units::meters(0.1);
    auto far = units::meters(100.0);

    auto perspective_matrix = geometry::perspective(fov, aspect_ratio, near, far);

Transformation Functions
------------------------

SQUINT provides functions to apply common transformations to 4x4 matrices.

Translation
^^^^^^^^^^^

The `translate` function applies a translation to a transformation matrix:

.. code-block:: cpp

    template <transformation_matrix T, typename U>
    void translate(T &matrix, const tensor<length_t<U>, shape<3>> &x, 
                   length_t<U> unit_length = length_t<U>{1});

This function modifies the input transformation matrix by applying a translation.

Example usage:

.. code-block:: cpp

    mat4 model_matrix = mat4::eye();
    vec3 translation{units::meters(2.0), units::meters(3.0), units::meters(-1.0)};

    geometry::translate(model_matrix, translation);

Rotation
^^^^^^^^

The `rotate` function applies a rotation to a transformation matrix:

.. code-block:: cpp

    template <transformation_matrix T, dimensionless_scalar U>
    void rotate(T &matrix, U angle, const tensor<U, shape<3>> &axis);

This function modifies the input transformation matrix by applying a rotation around an arbitrary axis.

Example usage:

.. code-block:: cpp

    mat4 model_matrix = mat4::eye();
    float angle = pi<float> / 4.0f; // 45 degrees
    vec3 axis{0.0f, 1.0f, 0.0f}; // Rotate around Y-axis

    geometry::rotate(model_matrix, angle, axis);

Scaling
^^^^^^^

The `scale` function applies a scale transformation to a transformation matrix:

.. code-block:: cpp

    template <transformation_matrix T, dimensionless_scalar U>
    void scale(T &matrix, const tensor<U, shape<3>> &s);

This function modifies the input transformation matrix by applying a scale transformation.

Example usage:

.. code-block:: cpp

    mat4 model_matrix = mat4::eye();
    vec3 scale_factors{2.0f, 2.0f, 2.0f}; // Scale uniformly by 2

    geometry::scale(model_matrix, scale_factors);

Combining Transformations
-------------------------

You can combine multiple transformations by applying them sequentially to a matrix:

.. code-block:: cpp

    mat4 model_matrix = mat4::eye();

    // Translate
    vec3 translation{units::meters(2.0), units::meters(3.0), units::meters(-1.0)};
    geometry::translate(model_matrix, translation);

    // Rotate
    float angle = pi<float> / 4.0f;
    vec3 axis{0.0f, 1.0f, 0.0f};
    geometry::rotate(model_matrix, angle, axis);

    // Scale
    vec3 scale_factors{2.0f, 2.0f, 2.0f};
    geometry::scale(model_matrix, scale_factors);

    // The model_matrix now represents a combined transformation
