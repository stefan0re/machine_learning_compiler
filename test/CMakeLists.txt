# test directory CMakeLists.txt

add_executable(jiter_tests test_base_instructions.cpp)

target_link_libraries(jiter_tests PRIVATE jiter)

add_test(NAME JiterTests COMMAND jiter_tests)
