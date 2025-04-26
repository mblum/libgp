# Validates that version numbers follow semantic versioning
include(CMakeParseArguments)

function(validate_version)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "VERSION" "")
    
    if(NOT ARG_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        message(FATAL_ERROR "Version '${ARG_VERSION}' does not follow semantic versioning (major.minor.patch)")
    endif()
    
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" VERSION_MATCH "${ARG_VERSION}")
    set(MAJOR "${CMAKE_MATCH_1}")
    set(MINOR "${CMAKE_MATCH_2}")
    set(PATCH "${CMAKE_MATCH_3}")
    
    if(MAJOR EQUAL 0)
        message(STATUS "Package is still in initial development (version ${ARG_VERSION})")
    endif()
endfunction()