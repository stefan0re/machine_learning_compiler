add_executable(EncodingTests EncodingTests.cpp)
target_link_libraries(EncodingTests PRIVATE Catch2::Catch2WithMain TenGen)
target_include_directories(EncodingTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME EncodingTests COMMAND EncodingTests)
