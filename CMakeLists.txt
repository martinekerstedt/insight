# Insight library and a test executable
# Configuration
# Set cmake minimum version
cmake_minimum_required(VERSION 3.10)



# Project name, language, version
project(insight LANGUAGES CXX)

set(INSIGHT_MAJOR_VERSION 0)
set(INSIGHT_MINOR_VERSION 1)
set(INSIGHT_PATCH_VERSION 0)
set(INSIGHT_VERSION
  ${INSIGHT_MAJOR_VERSION}.${INSIGHT_MINOR_VERSION}.${INSIGHT_PATCH_VERSION})



# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# NOT WORKING
# # Number of threads to use when building
# include(ProcessorCount)
#
# ProcessorCount(NUMBER_OF_CORES)
#
# if (NUMBER_OF_CORES EQUAL 0)
#     # Common core count is atleast four
#     set(NUMBER_OF_CORES 4)
# endif()
#
# set(CMAKE_BUILD_PARALLEL_LEVEL "${NUMBER_OF_CORES}" CACHE STRING "Number of parallel jobs" FORCE)



# Path to this dir, used in subdirectories
set(TOP_LEVEL_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "TOP_LEVEL_DIR" FORCE)



# Default build is release, but build type should be set in the call to cmake
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()



# If debug, set debug macro and use verbose makefile
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)

if (BUILD_TYPE_UPPER MATCHES DEBUG)
    add_compile_definitions(INSIGHT_DEBUG)
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefile" FORCE)
endif()



# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-Og") # This seems to disable debug info in exe
set(CMAKE_CXX_FLAGS_RELEASE "-O3")



# Define target as a shared library
add_library(${PROJECT_NAME}
    SHARED
        src/insight.cpp
        src/NeuralNet/neuralnet.cpp
        src/NeuralNet/functions.cpp
        src/NeuralNet/matrix.cpp
        src/NeuralNet/vector.cpp
)



# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${TOP_LEVEL_DIR}/include
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${TOP_LEVEL_DIR}/src
)



# Link pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Threads::Threads
)



# Project options
option(ENABLE_TEST "Clones google test and builds tests" ON)
option(ENABLE_BENCHMARK "Clones google benchmark and builds bencmarks" ON)
option(BUILD_EXAMPLE "Build example program" ON)


if (ENABLE_TEST)
    enable_testing()
    add_subdirectory(extern/googletest)
    add_subdirectory(test)
endif()


if (ENABLE_BENCHMARK)
    add_subdirectory(extern/googlebenchmark)
    add_subdirectory(benchmark)
endif()


if (BUILD_EXAMPLE)
    add_subdirectory(examples/mnist)
endif()













































































