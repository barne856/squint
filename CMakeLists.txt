cmake_minimum_required(VERSION 3.28)

# BLAS/LAPACKE backend options
set(BLAS_BACKEND "OpenBLAS" CACHE STRING "BLAS/LAPACK backend to use (OpenBLAS, MKL)")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS OpenBLAS MKL)

set(PROJECT_LANGUAGES CXX)

if(BLAS_BACKEND STREQUAL "OpenBLAS")
  list(APPEND PROJECT_LANGUAGES C Fortran)
endif()

project(SQUINT VERSION 1.0.0 LANGUAGES ${PROJECT_LANGUAGES})
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

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
        # Check for MKL in common locations
        set(MKL_LOCATIONS
            "/usr/include/mkl"
            "/opt/intel/oneapi/mkl/latest"
            "$ENV{MKLROOT}"
        )
        
        foreach(location ${MKL_LOCATIONS})
            if(EXISTS "${location}")
                set(MKLROOT "${location}" CACHE PATH "Root directory of MKL installation" FORCE)
                break()
            endif()
        endforeach()
        
        if(NOT MKLROOT)
            message(FATAL_ERROR "MKL not found. Please specify the correct MKLROOT.")
        endif()
        
        # Set include directories
        target_include_directories(SQUINT INTERFACE ${MKLROOT}/include)
        
        # Set library directories
        if(EXISTS "${MKLROOT}/lib/intel64")
            target_link_directories(SQUINT INTERFACE ${MKLROOT}/lib/intel64)
        else()
            target_link_directories(SQUINT INTERFACE /usr/lib/x86_64-linux-gnu)
        endif()
        
        # Link against MKL libraries
        target_link_libraries(SQUINT INTERFACE 
            -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
    else()
        # If found by FindMKL, use the provided targets
        target_link_libraries(SQUINT INTERFACE MKL::MKL)
    endif()
    
    # Add compile definition for MKL
    target_compile_definitions(SQUINT INTERFACE BLAS_BACKEND_MKL)
    
    # Set MKLROOT in the parent scope
    set(MKLROOT ${MKLROOT} PARENT_SCOPE)
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
    add_squint_test(tensor_tests)
    add_squint_test(linalg_tests)
    add_squint_test(geometry_tests)
endif()

add_library(SQUINT::SQUINT ALIAS SQUINT)
