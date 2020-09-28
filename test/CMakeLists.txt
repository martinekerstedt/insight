# Test executable for insight library
# Configuration
# Set cmake minimum version

# Project name
project(test_insight LANGUAGES CXX)


# Define output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)



# Define target
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/squaretest.cpp
    src/neuralnettest.cpp
    src/vectortest.cpp
    src/matrixtest.cpp
)



# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${TOP_LEVEL_DIR}/src
)



# Link, include paths/files are built-in to insight targe
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        insight
        gtest
)
