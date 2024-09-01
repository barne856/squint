Geometry Functions
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

To illustrate the difference between orthographic and perspective projections, consider the following diagram:

.. rst-class:: only-light

   .. tikz:: Orthographic vs Perspective Projection
        :libs: 3d
        :xscale: 80

        \begin{tikzpicture}[scale=0.7]
        \def\cubex{2}
        \def\cubey{2}
        \def\cubez{2}

        % Orthographic projection
        \begin{scope}[xshift=-5cm]
            \node[above] at (1,2.5,0) {Orthographic};
            \draw[black, thick] (0,0,0) -- (\cubex,0,0) -- (\cubex,\cubey,0) -- (0,\cubey,0) -- cycle;
            \draw[black, thick] (0,0,0) -- (0,0,\cubez) -- (\cubex,0,\cubez) -- (\cubex,0,0);
            \draw[black, thick] (0,0,\cubez) -- (0,\cubey,\cubez) -- (\cubex,\cubey,\cubez) -- (\cubex,0,\cubez);
            \draw[black, thick] (0,\cubey,0) -- (0,\cubey,\cubez);
            \draw[black, thick] (\cubex,\cubey,0) -- (\cubex,\cubey,\cubez);
        \end{scope}

        % Perspective projection
        \begin{scope}[xshift=5cm]
            \node[above] at (1,2.5,0) {Perspective};
            \def\vx{4}
            \def\vy{3}
            \pgfmathsetmacro{\vz}{6}

            % Define a scaling factor for the frustum effect
            \def\scale{0.5}

            % Draw the front face
            \draw[black, thick] (0,0,0) -- (\cubex,0,0) -- (\cubex,\cubey,0) -- (0,\cubey,0) -- cycle;

            % Draw the back face (scaled down for frustum effect)
            \draw[black, thick] 
                ({0*\scale},0,\cubez) -- ({\cubex*\scale},0,\cubez) -- 
                ({\cubex*\scale},{\cubey*\scale},\cubez) -- ({0*\scale},{\cubey*\scale},\cubez) -- cycle;

            % Connect front and back faces
            \draw[black, thick] (0,0,0) -- ({0*\scale},0,\cubez);
            \draw[black, thick] (\cubex,0,0) -- ({\cubex*\scale},0,\cubez);
            \draw[black, thick] (\cubex,\cubey,0) -- ({\cubex*\scale},{\cubey*\scale},\cubez);
            \draw[black, thick] (0,\cubey,0) -- ({0*\scale},{\cubey*\scale},\cubez);


        \end{scope}
        \end{tikzpicture}

.. rst-class:: only-dark

   .. tikz:: Orthographic vs Perspective Projection
        :libs: 3d
        :xscale: 80

        \begin{tikzpicture}[scale=0.7]
        \def\cubex{2}
        \def\cubey{2}
        \def\cubez{2}

        % Orthographic projection
        \begin{scope}[xshift=-5cm]
            \node[above, white] at (1,2.5,0) {Orthographic};
            \draw[white, thick] (0,0,0) -- (\cubex,0,0) -- (\cubex,\cubey,0) -- (0,\cubey,0) -- cycle;
            \draw[white, thick] (0,0,0) -- (0,0,\cubez) -- (\cubex,0,\cubez) -- (\cubex,0,0);
            \draw[white, thick] (0,0,\cubez) -- (0,\cubey,\cubez) -- (\cubex,\cubey,\cubez) -- (\cubex,0,\cubez);
            \draw[white, thick] (0,\cubey,0) -- (0,\cubey,\cubez);
            \draw[white, thick] (\cubex,\cubey,0) -- (\cubex,\cubey,\cubez);
        \end{scope}

        % Perspective projection
        \begin{scope}[xshift=5cm]
            \node[above, white] at (1,2.5,0) {Perspective};
            \def\vx{4}
            \def\vy{3}
            \pgfmathsetmacro{\vz}{6}

            % Define a scaling factor for the frustum effect
            \def\scale{0.5}

            % Draw the front face
            \draw[white, thick] (0,0,0) -- (\cubex,0,0) -- (\cubex,\cubey,0) -- (0,\cubey,0) -- cycle;

            % Draw the back face (scaled down for frustum effect)
            \draw[white, thick] 
                ({0*\scale},0,\cubez) -- ({\cubex*\scale},0,\cubez) -- 
                ({\cubex*\scale},{\cubey*\scale},\cubez) -- ({0*\scale},{\cubey*\scale},\cubez) -- cycle;

            % Connect front and back faces
            \draw[white, thick] (0,0,0) -- ({0*\scale},0,\cubez);
            \draw[white, thick] (\cubex,0,0) -- ({\cubex*\scale},0,\cubez);
            \draw[white, thick] (\cubex,\cubey,0) -- ({\cubex*\scale},{\cubey*\scale},\cubez);
            \draw[white, thick] (0,\cubey,0) -- ({0*\scale},{\cubey*\scale},\cubez);


        \end{scope}
        \end{tikzpicture}

This diagram shows the difference between orthographic and perspective projections. In the orthographic projection, parallel lines remain parallel. In the perspective projection, parallel lines converge towards a vanishing point, creating a sense of depth and distance.


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

To visualize the basic transformation functions (translation, rotation, and scaling), consider the following diagram:

.. rst-class:: only-light

   .. tikz:: Basic Transformations
      :libs: arrows
      :xscale: 80

      \begin{tikzpicture}[scale=0.7]
        % Original shape
        \begin{scope}[xshift=-6cm]
          \draw[thick, blue] (0,0) rectangle (2,1);
          \node[below] at (1,0) {Original};
        \end{scope}
        
        % Translation
        \begin{scope}[xshift=-2cm]
          \draw[thick, blue] (0,0) rectangle (2,1);
          \draw[thick, red] (1,1) rectangle (3,2);
          \draw[-{Stealth[length=3mm]}, thick] (1,0.5) -- (2,1.5);
          \node[below] at (1.5,0) {Translation};
        \end{scope}
        
        % Rotation
        \begin{scope}[xshift=2cm]
          \draw[thick, blue] (0,0) rectangle (2,1);
          \draw[thick, red, rotate around={45:(1,0.5)}] (0,0) rectangle (2,1);
          \draw[-{Stealth[length=3mm]}, thick, rotate around={22.5:(1,0.5)}] (1,0.5) arc (0:45:0.7);
          \node[below] at (1,0) {Rotation};
        \end{scope}
        
        % Scaling
        \begin{scope}[xshift=6cm]
          \draw[thick, blue] (0,0) rectangle (2,1);
          \draw[thick, red] (-0.5,-0.25) rectangle (2.5,1.25);
          \draw[-{Stealth[length=3mm]}, thick] (0,0) -- (-0.5,-0.25);
          \draw[-{Stealth[length=3mm]}, thick] (2,1) -- (2.5,1.25);
          \node[below] at (1,0) {Scaling};
        \end{scope}
      \end{tikzpicture}

.. rst-class:: only-dark

   .. tikz:: Basic Transformations
      :libs: arrows.meta
      :xscale: 80

      \begin{tikzpicture}[scale=0.7]
        % Original shape
        \begin{scope}[xshift=-6cm]
          \draw[thick, cyan] (0,0) rectangle (2,1);
          \node[below, text=white] at (1,0) {Original};
        \end{scope}
        
        % Translation
        \begin{scope}[xshift=-2cm]
          \draw[thick, cyan] (0,0) rectangle (2,1);
          \draw[thick, red] (1,1) rectangle (3,2);
          \draw[-{Stealth[length=3mm]}, thick, white] (1,0.5) -- (2,1.5);
          \node[below, text=white] at (1.5,0) {Translation};
        \end{scope}
        
        % Rotation
        \begin{scope}[xshift=2cm]
          \draw[thick, cyan] (0,0) rectangle (2,1);
          \draw[thick, red, rotate around={45:(1,0.5)}] (0,0) rectangle (2,1);
          \draw[-{Stealth[length=3mm]}, thick, white, rotate around={22.5:(1,0.5)}] (1,0.5) arc (0:45:0.7);
          \node[below, text=white] at (1,0) {Rotation};
        \end{scope}
        
        % Scaling
        \begin{scope}[xshift=6cm]
          \draw[thick, cyan] (0,0) rectangle (2,1);
          \draw[thick, red] (-0.5,-0.25) rectangle (2.5,1.25);
          \draw[-{Stealth[length=3mm]}, thick, white] (0,0) -- (-0.5,-0.25);
          \draw[-{Stealth[length=3mm]}, thick, white] (2,1) -- (2.5,1.25);
          \node[below, text=white] at (1,0) {Scaling};
        \end{scope}
      \end{tikzpicture}

This diagram illustrates the three basic transformations: translation (moving the object), rotation (turning the object around a point), and scaling (changing the size of the object).
