# Insight library and a test executable
# Configuration
# Set cmake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(insight VERSION 0.1 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)



# Define targets
# Library
add_library(${PROJECT_NAME}
    SHARED
        src/insight.cpp
        src/NeuralNet/neuralnet.cpp
        src/NeuralNet/neuron.cpp
)



# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)



# Make gtest
#ADD_SUBDIRECTORY(extern/gtest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

ADD_SUBDIRECTORY(extern/googletest)

set_target_properties(gtest gtest_main gmock gmock_main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/bin"
)



# Make test executable
ADD_SUBDIRECTORY(test)



