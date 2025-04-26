# Created by Manuel Blum on 2011-05-25.
# Copyright 2013 University of Freiburg.

cmake_minimum_required(VERSION 3.14)

project(libgp
    VERSION 0.1.4
    DESCRIPTION "Gaussian Process Regression Library"
    LANGUAGES CXX)

# Add cmake modules directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Load common settings
include(CommonSettings)

# Check for required C++ features
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Feature tests
check_cxx_source_compiles("
    #include <type_traits>
    int main() {
        static_assert(std::is_same_v<int, int>);
        return 0;
    }"
    HAVE_CPP17_FEATURES
)

if(NOT HAVE_CPP17_FEATURES)
    message(FATAL_ERROR "Compiler does not support required C++17 features")
endif()

# Set visibility settings
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Find dependencies
set(CMAKE_PREFIX_PATH "/opt/homebrew/share/eigen3/cmake" ${CMAKE_PREFIX_PATH})
find_package(Eigen3 3.0.1 NO_MODULE REQUIRED)
find_package(Threads REQUIRED)

# Create library target
add_library(gp)
add_library(libgp::gp ALIAS gp)

# Include sources
include("Sources.cmake")
target_sources(gp PRIVATE ${LIBGP_SRC})

# Apply common settings
libgp_set_common_properties(gp)

# Modern way to handle include directories
target_include_directories(gp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gp
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        Threads::Threads
)

target_compile_definitions(gp
    PRIVATE
        libgp_EXPORTS
        $<$<BOOL:${BUILD_TESTING}>:ENABLE_TESTING>
)

# Enable testing with CTest
if(BUILD_TESTS)
    enable_testing()
    include(CTest)
    add_subdirectory(tests ${CMAKE_CURRENT_BINARY_DIR}/tests_build)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libgp)

# Configure version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure config files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libgp-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfig.cmake"
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(TARGETS gp
    EXPORT libgp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gp
    FILES_MATCHING PATTERN "*.h"
)

# Install CMake config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libgpConfigVersion.cmake"
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Export targets
install(EXPORT libgp-targets
    FILE libgpTargets.cmake
    NAMESPACE libgp::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Generate pkg-config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/contrib/libgp.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/libgp.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgp.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Add components
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
