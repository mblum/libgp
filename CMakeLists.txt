cmake_minimum_required(VERSION 3.14...3.22)

project(libgp
    VERSION "0.3.0"
    DESCRIPTION "C++ Library for Gaussian Process Regression"
    LANGUAGES CXX)

include(FetchContent)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(LIBGP_BUILD_ALL "Build everything (tests and examples)" OFF)
option(LIBGP_BUILD_TESTS "Build libgp tests" ${LIBGP_BUILD_ALL})
option(LIBGP_BUILD_EXAMPLES "Build example applications" ${LIBGP_BUILD_ALL})
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)  # Changed to OFF for static linking

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Added for Python bindings

# Dependencies
find_package(Eigen3 REQUIRED)

# Generate version header
configure_file(
    "${PROJECT_SOURCE_DIR}/include/gp_version.h.in"
    "${PROJECT_BINARY_DIR}/include/gp_version.h"
    @ONLY
)

# Core library
add_library(gp STATIC  # Changed to STATIC
    src/gp.cc
    src/gp_utils.cc
    src/sampleset.cc
    src/rprop.cc
    src/cg.cc
    src/input_dim_filter.cc
    src/cov.cc
    src/cov_factory.cc
    src/cov_linear_ard.cc
    src/cov_linear_one.cc
    src/cov_matern3_iso.cc
    src/cov_matern5_iso.cc
    src/cov_noise.cc
    src/cov_periodic.cc
    src/cov_periodic_matern3_iso.cc
    src/cov_prod.cc
    src/cov_rq_iso.cc
    src/cov_se_ard.cc
    src/cov_se_iso.cc
    src/cov_sum.cc
)

target_include_directories(gp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gp PUBLIC Eigen3::Eigen)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    
    # Create the Python module with the correct name and location
    pybind11_add_module(libgp_cpp src/bindings.cpp)
    target_link_libraries(libgp_cpp PRIVATE gp Eigen3::Eigen)
    set_target_properties(libgp_cpp PROPERTIES
        OUTPUT_NAME "libgp_cpp"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libgp"
    )
    
    # Install Python module and package files
    install(TARGETS libgp_cpp
        LIBRARY DESTINATION "libgp"
        ARCHIVE DESTINATION "libgp"
    )
    
    # Install Python package files
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/libgp/__init__.py"
            DESTINATION "libgp")
endif()

# Tests
if(LIBGP_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    function(add_gp_test name)
        add_executable(${name} tests/${name}.cc)
        target_link_libraries(${name} PRIVATE gp GTest::GTest GTest::Main)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_gp_test(test_gp_regression)
    add_gp_test(test_log_likelihood)
    add_gp_test(test_cov_factory)
    add_gp_test(test_covariance_functions)
    add_gp_test(test_gp_utils)
    add_gp_test(test_optimizer)
endif()

# Examples
if(LIBGP_BUILD_EXAMPLES)
    add_executable(gp_example_dense examples/gp_example_dense.cc)
    target_link_libraries(gp_example_dense PRIVATE gp)
endif()

# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libgp)

install(TARGETS gp
    EXPORT libgp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/gp_version.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT libgp-targets
    FILE libgpTargets.cmake
    NAMESPACE libgp::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create and install config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/contrib/libgp.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfig.cmake"
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfigVersion.cmake"
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Generate and install pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/contrib/libgp.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libgp.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libgp.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)