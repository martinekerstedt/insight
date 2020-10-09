# Google Test library


# Include module
include(FetchContent)



# Edit FetchContent settings to keep googletest repo in extern/googletest
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE STRING "" FORCE)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)



# NOTE: A git version of 1.6.5 or later is required if this download method is used.
# Declare details about the download
# Commit hash to a commit pushed on Jun 26 2020, "Googletest export".
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG aee0f9d9b5b87796ee8a0ab26b7587ec30e8858e
    GIT_CONFIG advice.detachedHead=false
    GIT_PROGRESS ON
)



# Set options before running googletest cmake
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)



# Clones googletest repo and runs it's CMakeLists.txt
FetchContent_MakeAvailable(googletest)



# Change the output directories for targets defined by googletest
set_target_properties(
    gtest gtest_main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib"
)

if (BUILD_GMOCK)

    set_target_properties(
        gmock gmock_main
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib"
    )

endif()






















































































