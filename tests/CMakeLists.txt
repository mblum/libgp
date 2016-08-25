FIND_PACKAGE(Threads REQUIRED)

# ----- Testing -----
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest/include)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest-build)

# ----- Add sources -----
INCLUDE("Sources.cmake")

ADD_EXECUTABLE(gptest ${LIBGP_TESTS})
TARGET_LINK_LIBRARIES(gptest gp ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main)
