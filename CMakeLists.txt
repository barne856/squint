cmake_minimum_required(VERSION 3.28)

## Project Settings
project(SQUINT)
set(CMAKE_CXX_STANDARD 23)
include_directories("./include")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

## Create the test executable
add_executable(tests tests/tests.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

## Discover tests
include(CTest)
include(Catch)
catch_discover_tests(tests)

## Create the sample executable
add_executable(sample1 tests/samples/sample1.cpp)
