# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg. All rights reserved.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)
PROJECT(libgp CXX)

SET( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Doxygen)
FIND_PACKAGE(Eigen3 REQUIRED)

# ----- Add include directories -----
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- Specify source files -----
SET(LIB_SRC src/cov.cc src/cov_noise.cc src/cov_se_ard.cc src/cov_se_iso.cc src/cov_sum.cc src/gp.cc)

# ----- Specify header files -----
SET(LIB_INTERFACES include/cov.h include/cov_factory.h include/cov_noise.h include/cov_se_ard.h
    include/cov_se_iso.h include/cov_sum.h include/gp.h include/pattern.h include/doxygen.h)

# ----- Add library target -----
ADD_LIBRARY(gp STATIC ${LIB_SRC} ${LIB_INTERFACES})

ADD_EXECUTABLE(gp_example examples/gp_example.cc)
TARGET_LINK_LIBRARIES(gp_example gp)

# ----- Testing -----
#ENABLE_TESTING()
#ADD_TEST(gp_example )

# ----- Add custom doxygen target -----
IF(DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET(libgp_doc COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE} 
	                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

ENDIF()

