cmake_minimum_required(VERSION 3.28)

## Project Settings
project(SQUINT)
set(CMAKE_CXX_STANDARD 23)
include_directories("./include")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define an INTERFACE library target
add_library(squint INTERFACE)

# Add include directories
target_include_directories(squint
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

## Find or fetch Catch2
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later version
)

# Disable Catch2 self-tests
set(BUILD_TESTING OFF)

FetchContent_MakeAvailable(Catch2)

# # TODO, tests are not working on MSVC, but work with clang
# ## Create the test executable
# add_executable(tests tests/tests.cpp)
# target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
# 
# ## Discover tests
# include(CTest)
# include(Catch)
# catch_discover_tests(tests)

# Create the sample executable
add_executable(sample2 tests/samples/sample2.cpp)
add_executable(sample1 tests/samples/sample1.cpp)
