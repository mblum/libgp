INCLUDE(ExternalProject)

# ----- Download and build gtest -----
ExternalProject_Add(googletest
  SVN_REPOSITORY "http://googletest.googlecode.com/svn/tags/release-1.6.0"
  INSTALL_COMMAND ""
)

# ----- Testing -----
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest/include)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)

# ----- Add sources -----
INCLUDE("Sources.cmake")

ADD_EXECUTABLE(gptest ${LIBGP_TESTS})
TARGET_LINK_LIBRARIES(gptest gp gtest gtest_main)
