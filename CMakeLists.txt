cmake_minimum_required(VERSION 3.28)

# BLAS/LAPACKE backend options
set(SQUINT_BLAS_BACKEND "NONE" CACHE STRING "BLAS/LAPACK backend to use (OpenBLAS, MKL, NONE)")
set_property(CACHE SQUINT_BLAS_BACKEND PROPERTY STRINGS OpenBLAS MKL NONE)

set(PROJECT_LANGUAGES CXX)
if(SQUINT_BLAS_BACKEND STREQUAL "OpenBLAS")
  list(APPEND PROJECT_LANGUAGES C Fortran)
endif()

project(SQUINT VERSION 1.0.0 LANGUAGES ${PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Function to setup BLAS backend
function(setup_blas_backend)
  if(SQUINT_BLAS_BACKEND STREQUAL "OpenBLAS")
    setup_openblas()
  elseif(SQUINT_BLAS_BACKEND STREQUAL "MKL")
    setup_mkl()
  elseif(SQUINT_BLAS_BACKEND STREQUAL "NONE")
    setup_none()
  else()
    message(FATAL_ERROR "Unsupported BLAS backend: ${SQUINT_BLAS_BACKEND}")
  endif()
endfunction()

# Function to setup OpenBLAS
function(setup_openblas)
  FetchContent_Declare(
    openblas
    GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
    GIT_TAG develop
  )
  set(USE_THREAD ON)
  option(BUILD_TESTING OFF)
  FetchContent_MakeAvailable(openblas)
  target_link_libraries(SQUINT INTERFACE openblas)
  target_include_directories(SQUINT INTERFACE
    ${openblas_SOURCE_DIR}/lapack-netlib/LAPACKE/include
    ${CMAKE_BINARY_DIR}/generated
    ${CMAKE_BINARY_DIR}
  )
  target_compile_definitions(SQUINT INTERFACE SQUINT_BLAS_BACKEND_OPENBLAS)
  set(EXCLUDE_FROM_ALL TRUE)
endfunction()

# Function to setup MKL
function(setup_mkl)
  # Try to find MKL using CMake's built-in FindMKL module
  find_package(MKL REQUIRED)
  # If found by FindMKL, use the provided targets
  target_link_libraries(SQUINT INTERFACE MKL::MKL)
  # Add compile definition for MKL
  target_compile_definitions(SQUINT INTERFACE SQUINT_BLAS_BACKEND_MKL)
endfunction()

# Function to setup NONE (fallback implementations)
function(setup_none)
  target_compile_definitions(SQUINT INTERFACE SQUINT_BLAS_BACKEND_NONE)
endfunction()

# Option to control building tests
option(SQUINT_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})

# Create interface library for SQUINT
add_library(SQUINT INTERFACE)
target_include_directories(SQUINT INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Setup BLAS backend
setup_blas_backend()

# Tests
if(SQUINT_BUILD_TESTS)
  # Download doctest header
  file(DOWNLOAD
    https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h
    ${CMAKE_BINARY_DIR}/doctest.h
  )
  # Create interface library for doctest
  add_library(doctest INTERFACE)
  target_include_directories(doctest INTERFACE ${CMAKE_BINARY_DIR})
  enable_testing()

  # Function to create test targets
  function(add_squint_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE SQUINT doctest)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "SQUINT")
  endfunction()

  # Add tests
  add_squint_test(quantity_tests)
  add_squint_test(dimension_tests)
  add_squint_test(unit_tests)
  add_squint_test(math_tests)
  add_squint_test(tensor_tests)
  # add_squint_test(linalg_tests)
  # add_squint_test(geometry_tests)
endif()

add_library(SQUINT::SQUINT ALIAS SQUINT)

# Documentation
option(SQUINT_BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${PROJECT_IS_TOP_LEVEL})

if(SQUINT_BUILD_DOCUMENTATION)
  # Find Doxygen
  find_package(Doxygen REQUIRED)
  
  # Find Sphinx
  find_package(Sphinx REQUIRED)

  # Find all the public headers
  get_target_property(SQUINT_PUBLIC_HEADER_DIR SQUINT INTERFACE_INCLUDE_DIRECTORIES)
  file(GLOB_RECURSE SQUINT_PUBLIC_HEADERS ${SQUINT_PUBLIC_HEADER_DIR}/*.h)

  set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
  set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # Replace variables inside @@ with the current values
  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  # Doxygen won't create this for us
  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

  # Only regenerate Doxygen when the Doxyfile or public headers change
  add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                     DEPENDS ${SQUINT_PUBLIC_HEADERS}
                     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                     MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                     COMMENT "Generating API documentation with Doxygen"
                     VERBATIM)

  # Nice named target so we can run the job easily
  add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

  set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs)
  set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
  set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

  # Only regenerate Sphinx when:
  # - Doxygen has rerun
  # - Our doc files have been updated
  # - The Sphinx config has been updated
  add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                     COMMAND 
                       ${SPHINX_EXECUTABLE} -b html
                       # Tell Breathe where to find the Doxygen output
                       -Dbreathe_projects.SQUINT=${DOXYGEN_OUTPUT_DIR}/xml
                     ${SPHINX_SOURCE} ${SPHINX_BUILD}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     DEPENDS
                     # Other docs files you want to track should go here (or in some variable)
                     ${CMAKE_CURRENT_SOURCE_DIR}/docs/index.rst
                     ${DOXYGEN_INDEX_FILE}
                     MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                     COMMENT "Generating documentation with Sphinx")

  # Nice named target so we can run the job easily
  add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

  # Add an install target to install the docs
  include(GNUInstallDirs)
  install(DIRECTORY ${SPHINX_BUILD}
    DESTINATION ${CMAKE_INSTALL_DOCDIR})

endif()
include_directories(include)
add_executable(main main.cpp)
target_link_libraries(main SQUINT)