# Test executable for insight library
# Configuration
# Set cmake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(test_insight VERSION 0.1 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)



# Define target
add_executable(${PROJECT_NAME}
    src/main.cpp
)



# Link, include paths/files are built-in to insight targe
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        insight
        gtest
)


##################################
# Just make the test runnable with
#   $ make test

#enable_testing()
#add_test(NAME    ${PROJECT_NAME} 
#         COMMAND ${PROJECT_NAME}
#)
