pybind11_add_module(pygp MODULE bindings.cpp)
target_link_libraries(pygp PRIVATE libgp::gp)

# Add include directories
target_include_directories(pygp PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install the Python module
install(TARGETS pygp 
    LIBRARY DESTINATION ${Python_SITEARCH}
    RUNTIME DESTINATION ${Python_SITEARCH}
)