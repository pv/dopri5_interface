cmake_minimum_required (VERSION 3.0)
project (dopri5_interface)
enable_language (Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories (.)

add_executable(example example.cpp dopri5_nocommon.f dopri5_interface.h)
target_link_libraries(example Eigen3::Eigen)
target_compile_definitions(example PRIVATE USE_EIGEN)

add_executable(example_original example.cpp dopri5.f dopri5_interface.h)
target_link_libraries(example_original Eigen3::Eigen)
target_compile_definitions(example_original PRIVATE DOPRI5_INTERACE_USE_ORIGINAL USE_EIGEN)

find_package(Doxygen)
option(BUILD_DOCS "Build documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if (BUILD_DOCS)
  set (DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set (DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file (${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  add_custom_target (doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif (BUILD_DOCS)
