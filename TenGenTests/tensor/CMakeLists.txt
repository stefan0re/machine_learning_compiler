add_executable(TensorTests TensorTests.cpp)
target_link_libraries(TensorTests PRIVATE Catch2::Catch2WithMain TenGen xtensor xtl)
target_include_directories(TensorTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME TensorTests COMMAND TensorTests)
