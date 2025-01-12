cmake_minimum_required(VERSION 3.20)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/simulation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/util.hpp)
add_executable(euler_fluid ${module_source_files})
set_generic_compiler_options(euler_fluid PRIVATE)
target_link_libraries(euler_fluid PRIVATE klgl)
target_include_directories(euler_fluid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(euler_fluid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)

find_package(TBB REQUIRED COMPONENTS tbb)
target_link_libraries(euler_fluid PRIVATE tbb)
