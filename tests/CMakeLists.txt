include(FetchContent)

# Set minimum CMake version for Google Test
set(CMAKE_MINIMUM_REQUIRED_VERSION 3.14)

# Modern GTest integration using FetchContent
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
include(GoogleTest)
enable_testing()

# Include test sources
include("Sources.cmake")

# Create test executable
add_executable(gptest ${LIBGP_TESTS})
target_link_libraries(gptest
    PRIVATE
        gp
        GTest::gtest
        GTest::gtest_main
        ${CMAKE_THREAD_LIBS_INIT}
)

# Apply common settings
libgp_set_common_properties(gptest)

# Add test labels based on file names
foreach(TEST_FILE ${LIBGP_TESTS})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    if(${TEST_NAME} MATCHES "^test_")
        string(SUBSTRING ${TEST_NAME} 5 -1 LABEL_NAME)
    else()
        set(LABEL_NAME ${TEST_NAME})
    endif()
    set_source_files_properties(${TEST_FILE} PROPERTIES
        LABELS "${LABEL_NAME}"
    )
endforeach()

# Configure test discovery with increased timeout
gtest_discover_tests(gptest
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
    DISCOVERY_TIMEOUT 60
    XML_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test-results"
)
