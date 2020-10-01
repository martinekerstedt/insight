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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define output directories
set(TOP_LEVEL_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "TOP_LEVEL_DIR" FORCE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TOP_LEVEL_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TOP_LEVEL_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TOP_LEVEL_DIR}/bin)


#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
#set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)


# Define targets
# Library
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
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)



# Make googletest
ADD_SUBDIRECTORY(extern/googletest)



# Make test executable
ADD_SUBDIRECTORY(test)



# Make example executable
ADD_SUBDIRECTORY(examples)













































































