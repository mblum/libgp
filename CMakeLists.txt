# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)
PROJECT(libgp CXX C)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_BUILD_TYPE "Release")

FIND_PACKAGE(Eigen3 3.0 REQUIRED)
FIND_PACKAGE(GTest)

# ----- Add include directories -----
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- Add sources -----
INCLUDE("Sources.cmake")

# ----- Add library target -----
ADD_LIBRARY(gp ${LIBGP_SRC} ${LIBGP_INTERFACES})

# ----- Add example targets -----
ADD_EXECUTABLE(gpdense examples/gp_example_dense.cc)
# ADD_EXECUTABLE(gpsparse examples/gp_example_sparse.cc)
TARGET_LINK_LIBRARIES(gpdense gp)
# TARGET_LINK_LIBRARIES(gpsparse gp)

# ----- Testing -----
IF (GTEST_FOUND)
  ENABLE_TESTING()
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
  ADD_EXECUTABLE(gptest ${LIBGP_TESTS})
  TARGET_LINK_LIBRARIES(gptest gp ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
  ADD_TEST(gptest ${CMAKE_CURRENT_BINARY_DIR}/gptest)
ENDIF ()

# ----- Build type -----
SET(CMAKE_BUILD_TYPE Release)

# ----- Install -----
INSTALL(FILES ${LIB_INTERFACES} DESTINATION include)
INSTALL(TARGETS gp
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

