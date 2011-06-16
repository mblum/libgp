# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)
PROJECT(libgp CXX)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(GTest)

# ----- Add include directories -----
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES("/opt/local/include/ufsparse")

# ----- Specify source files -----
SET(LIB_SRC src/gp.cc src/gp_sparse.cc src/cov.cc src/cov_noise.cc 
    src/cov_se_ard.cc src/cov_se_iso.cc src/cov_se_iso_compact.cc  
    src/cov_sum.cc src/cov_matern3_iso.cc src/cov_matern5_iso.cc 
    src/cov_rq_iso.cc src/gp_utils.cc)

# ----- Specify header files -----
SET(LIB_INTERFACES include/gp.h include/gp_sparse.h include/sampleset.h 
    include/cov.h include/cov_factory.h include/cov_noise.h 
    include/cov_se_ard.h include/cov_se_iso.h include/cov_se_iso_compact.h  
    include/cov_sum.h include/cov_matern3_iso.h include/cov_matern5_iso.h 
    include/cov_rq_iso.h include/doxygen.h include/gp_utils.h)

# ----- Specify test files -----
SET(LIB_TESTS tests/cov_test.cc tests/gp_regression_test.cc tests/gp_utils_test.cc)

# ----- Add library target -----
ADD_LIBRARY(gp STATIC ${LIB_SRC} ${LIB_INTERFACES})
TARGET_LINK_LIBRARIES(gp "-lamd -lcamd -lccolamd -lcholmod -lcolamd -lcxsparse -lblas -lmetis")

ADD_EXECUTABLE(gp_example examples/gp_example.cc)
TARGET_LINK_LIBRARIES(gp_example gp)

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

