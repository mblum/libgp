# Common settings for all targets in libgp project

# Set C++ standard
macro(libgp_set_cpp_standard target)
    set_target_properties(${target}
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
endmacro()

# Set common compiler warnings
macro(libgp_set_compiler_warnings target)
    target_compile_options(${target}
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endmacro()

# Set common build settings
macro(libgp_set_common_properties target)
    libgp_set_cpp_standard(${target})
    libgp_set_compiler_warnings(${target})
endmacro()