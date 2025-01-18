find_package(TBB REQUIRED COMPONENTS tbb)
target_link_libraries(euler_fluid PRIVATE tbb)
