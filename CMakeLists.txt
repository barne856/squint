cmake_minimum_required(VERSION 3.28)

project(SQUINT VERSION 1.0.0 LANGUAGES CXX C Fortran)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# BLAS/LAPACKE backend options
set(BLAS_BACKEND "OpenBLAS" CACHE STRING "BLAS/LAPACK backend to use (OpenBLAS, MKL)")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS OpenBLAS MKL)

# Function to setup BLAS backend
function(setup_blas_backend)
    if(BLAS_BACKEND STREQUAL "OpenBLAS")
        setup_openblas()
    elseif(BLAS_BACKEND STREQUAL "MKL")
        setup_mkl()
    else()
        message(FATAL_ERROR "Unsupported BLAS backend: ${BLAS_BACKEND}")
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
    target_compile_definitions(SQUINT INTERFACE BLAS_BACKEND_OPENBLAS)

    set(EXCLUDE_FROM_ALL TRUE)
endfunction()

# Function to setup MKL
function(setup_mkl)
    # First, try to find MKL using CMake's built-in FindMKL module
    find_package(MKL QUIET)
    
    if(NOT MKL_FOUND)
        # If not found, look for MKL in the default apt installation directory
        set(MKLROOT "/usr/include/mkl" CACHE PATH "Root directory of MKL installation")
        
        if(NOT EXISTS ${MKLROOT})
            message(FATAL_ERROR "MKL not found. Please specify the correct MKLROOT.")
        endif()
        
        # Set include directories
        target_include_directories(SQUINT INTERFACE ${MKLROOT})
        
        # Set library directories
        target_link_directories(SQUINT INTERFACE /usr/lib/x86_64-linux-gnu)
        
        # Link against MKL libraries
        target_link_libraries(SQUINT INTERFACE 
            -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
    else()
        # If found by FindMKL, use the provided targets
        target_link_libraries(SQUINT INTERFACE MKL::MKL)
    endif()
    
    # Add compile definition for MKL
    target_compile_definitions(SQUINT INTERFACE BLAS_BACKEND_MKL)
endfunction()

# Option to control building tests
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(SQUINT_BUILD_TESTS "Build the tests" ON)
else()
    option(SQUINT_BUILD_TESTS "Build the tests" OFF)
endif()

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
    add_squint_test(tensor_tests)
endif()

# Main executable (temporary testing purposes)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE SQUINT)
endif()