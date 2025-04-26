# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

# Examples configuration
add_executable(gpdense gp_example_dense.cc)
target_link_libraries(gpdense
    PRIVATE
        libgp::gp
)

# Apply common settings
libgp_set_common_properties(gpdense)

# Install examples (optional)
install(TARGETS gpdense
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    OPTIONAL
)

# Install example source files for reference
install(FILES
    gp_example_dense.cc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/libgp/examples
    OPTIONAL
)
