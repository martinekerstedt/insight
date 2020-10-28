# Test executable for insight library
# Configuration
# Set cmake minimum version

# Project name
project(test_insight LANGUAGES CXX)


# Define output directories
string(REPLACE "${TOP_LEVEL_DIR}" "${CMAKE_BINARY_DIR}" OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})



# Define target
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/neuralnettest.cpp
    src/vectortest.cpp
    src/matrixtest.cpp
)



# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${TOP_LEVEL_DIR}/src
)



# Link, include paths/files are built-in to insight target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        insight
        gtest
)



# Register the executable as a test
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
