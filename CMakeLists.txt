# Insight library and a test executable
# Configuration
# Set cmake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(insight LANGUAGES CXX)

set(INSIGHT_MAJOR_VERSION 0)
set(INSIGHT_MINOR_VERSION 1)
set(INSIGHT_PATCH_VERSION 0)
set(INSIGHT_VERSION
  ${INSIGHT_MAJOR_VERSION}.${INSIGHT_MINOR_VERSION}.${INSIGHT_PATCH_VERSION})


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



# Make googletest
ADD_SUBDIRECTORY(extern/googletest)



# Make test executable
ADD_SUBDIRECTORY(test)














































































