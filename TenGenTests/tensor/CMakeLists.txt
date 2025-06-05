add_executable(TensorTests TensorTests.cpp)
target_link_libraries(TensorTests PRIVATE Catch2::Catch2WithMain TenGen)
target_include_directories(TensorTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME TensorTests COMMAND TensorTests)

add_executable(TensorXTensorTests TensorXTensorTests.cpp)
target_link_libraries(TensorXTensorTests PRIVATE Catch2::Catch2WithMain TenGen xtensor xtl xtensor-blas)
target_include_directories(TensorXTensorTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME TensorXTensorTests COMMAND TensorXTensorTests)