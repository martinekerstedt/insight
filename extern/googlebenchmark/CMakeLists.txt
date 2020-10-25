# Google Benchmark library



# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O0")



# Include module
include(FetchContent)



# Edit FetchContent settings to keep googlebenchmark repo in extern/googlebenchmark
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE STRING "" FORCE)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)



# NOTE: A git version of 1.6.5 or later is required if this download method is used.
# Declare details about the download
# The commit hash is a commit pushed on Sep 29 2020, "Add CycleTimer implementation for M68K architecture".
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG ffe1342eb2faa7d2e7c35b4db2ccf99fab81ec20
    GIT_CONFIG advice.detachedHead=false
    GIT_PROGRESS ON
)



# Set options before running googlebenchmark cmake
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "BENCHMARK_ENABLE_INSTALL" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "BENCHMARK_DOWNLOAD_DEPENDENCIES" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "BENCHMARK_ENABLE_TESTING" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "BENCHMARK_ENABLE_GTEST_TESTS" FORCE)
#set(GOOGLETEST_PATH "${TOP_LEVEL_DIR}/googletest/googletest-src" CACHE STRING "GOOGLETEST_PATH" FORCE)



# Clones googlebenchmark repo and runs it's CMakeLists.txt
FetchContent_MakeAvailable(googlebenchmark)



# Change the output directories for targets defined by googlebenchmark
string(REPLACE "${TOP_LEVEL_DIR}" "${CMAKE_BINARY_DIR}" OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}")

set_target_properties(
    benchmark benchmark_main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)








































































