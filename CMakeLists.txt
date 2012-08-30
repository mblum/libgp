# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg.

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE( configuration.cmake.in )
PROJECT(libgp CXX C)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})


FIND_PACKAGE(Eigen3 3.0.9 REQUIRED)

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
IF(BUILD_TESTS)
  ADD_SUBDIRECTORY(tests)
ENDIF()

# ----- Install -----
INSTALL(FILES ${LIB_INTERFACES} DESTINATION include)
INSTALL(TARGETS gp
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

