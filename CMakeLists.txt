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
ADD_SUBDIRECTORY(extern/gtest)

#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
#set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

#ADD_SUBDIRECTORY(extern/googletest)

#set_target_properties(gtest gtest_main gmock gmock_main
#    PROPERTIES
#    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/lib"
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/lib"
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/bin"
#)


#include(FetchContent)

#FetchContent_Declare(gtest_git
#    GIT_REPOSITORY https://github.com/google/googletest.git
#    GIT_TAG master
#)

#FetchContent_GetProperties(gtest_git)

#message(STATUS "gtest_git_POPULATED: ${gtest_git_POPULATED}\n")

#if(NOT gtest_git_POPULATED)
#    FetchContent_Populate(gtest_git)
#    
#    message(STATUS "gtest_git_SOURCE_DIR: ${gtest_git_SOURCE_DIR}\n")
#    message(STATUS "gtest_git_BINARY_DIR: ${gtest_git_BINARY_DIR}\n")
#    
#    add_subdirectory(${gtest_git_SOURCE_DIR} ${gtest_git_BINARY_DIR})
#endif()

#FetchContent_MakeAvailable(gtest_git)






# Make test executable
ADD_SUBDIRECTORY(test)














































































