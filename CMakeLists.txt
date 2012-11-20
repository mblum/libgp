# Created by Manuel Blum on 2011-05-25.
# Copyright 2011 University of Freiburg.

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# if no option is given, standard is release
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE "-DCLSQUARE -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-DCLSQUARE -Wall -g")

PROJECT(libgp CXX C)

OPTION(BUILD_TESTS "Build tests" ON)
OPTION(BUILD_EXAMPLES "Build examples" ON)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Eigen3 3.0.9 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python)
FIND_PACKAGE(PythonLibs)


# ----- Add include directories -----
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- Add sources -----
INCLUDE("Sources.cmake")

# ----- Add library target -----
#ADD_LIBRARY(gp STATIC ${LIBGP_SRC} ${LIBGP_INTERFACES})
ADD_LIBRARY(gp STATIC ${LIBGP_SRC} ${LIBGP_INTERFACES})

# ----- Add example targets -----
IF(BUILD_EXAMPLES)
  ADD_SUBDIRECTORY(examples)
ENDIF()

# ----- Testing -----
IF(BUILD_TESTS)
  ADD_SUBDIRECTORY(tests)
ENDIF()

# ----- Python bindings -----
IF(Boost_FOUND)
  #  SET(PYTHON_INCLUDE_DIRS /usr/local/Cellar/python/2.7.3/Frameworks/Python.framework/Headers)
  #  SET(PYTHON_LIBRARIES /usr/local/Cellar/python/2.7.3/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib)
  INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
  ADD_LIBRARY(gpy MODULE python/hello.cpp)
  TARGET_LINK_LIBRARIES(gpy ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
ENDIF()

# ----- Install -----
INSTALL(FILES ${LIB_INTERFACES} DESTINATION include)
INSTALL(TARGETS gp
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

