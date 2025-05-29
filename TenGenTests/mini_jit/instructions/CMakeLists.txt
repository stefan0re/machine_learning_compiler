add_executable(EncodingTests EncodingTests.cpp)
target_link_libraries(EncodingTests PRIVATE Catch2::Catch2WithMain)
add_test(NAME EncodingTests COMMAND EncodingTests)
