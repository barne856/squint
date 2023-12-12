cmake_minimum_required(VERSION 3.28)

## Project Settings
project(SQUINT)
set(CMAKE_CXX_STANDARD 23)
include_directories("./include")
set(BUILD_TESTING OFF) # disable Catch2 self tests
set(CMAKE_CXX_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -m64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++abi")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Include Catch2 for Unit Tests
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

## Create the tests
add_executable(tests tests/tests.cpp)
add_executable(sample tests/sample.cpp)
add_library(squint)
target_sources(squint PUBLIC FILE_SET CXX_MODULES FILES
  src/dimension.cpp
  src/linalg.cpp
  src/optimize.cpp
  src/quantity.cpp
  src/tensor.cpp
  src/squint.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain squint)
target_link_libraries(sample PRIVATE squint)
if(USE_MKL)
    # Set to your MKL directory path, you can source /opt/intel/oneapi/setvars.sh to set the environment variable MKLROOT
    # you probably need to source this or else you could get OpenMP not found errors
    set(MKL_DIR "$ENV{MKLROOT}/lib/cmake/mkl" CACHE PATH "Path to MKLConfig.cmake")
    add_compile_definitions(SQUINT_USE_MKL)
    add_compile_definitions(MKL_ILP64)
    find_package(MKL REQUIRED)
    target_compile_options(tests PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_compile_options(sample PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(tests PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(sample PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(tests PRIVATE $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(sample PRIVATE $<LINK_ONLY:MKL::MKL>)
endif()
catch_discover_tests(tests)
