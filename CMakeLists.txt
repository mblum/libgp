# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)
PROJECT(libgp CXX C)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(Cholmod REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(GTest)

ADD_DEFINITIONS("-DEIGEN_CHOLMOD_SUPPORT")

# ----- Add include directories -----
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDES})

INCLUDE("Sources.cmake")

# ----- Specify test files -----
SET(LIB_TESTS tests/cov_test.cc tests/gp_regression_test.cc tests/gp_utils_test.cc tests/gp_sparse_regression_test.cc)

# ----- Add library target -----
ADD_LIBRARY(gp ${LIB_SRC} ${LIB_INTERFACES})
TARGET_LINK_LIBRARIES(gp ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES})

# ----- Add example targets -----
ADD_EXECUTABLE(gpdense examples/gp_example_dense.cc)
ADD_EXECUTABLE(gpsparse examples/gp_example_sparse.cc)
TARGET_LINK_LIBRARIES(gpdense gp)
TARGET_LINK_LIBRARIES(gpsparse gp)

# ----- Testing -----
IF (GTEST_FOUND)
  ENABLE_TESTING()
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
  ADD_EXECUTABLE(gp_test ${LIB_TESTS})
  TARGET_LINK_LIBRARIES(gp_test gp ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
  ADD_TEST(gp_test ${CMAKE_CURRENT_BINARY_DIR}/gp_test)
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

