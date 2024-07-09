cmake_minimum_required(VERSION 3.28)

project(SQUINT VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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