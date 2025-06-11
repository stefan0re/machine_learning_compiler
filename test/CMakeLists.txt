cmake_minimum_required(VERSION 3.14)
project(mini_jit_tests LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
)
FetchContent_MakeAvailable(Catch2)

set(TEST_SOURCES
    mini_jit/test_instructions.cpp
    mini_jit/test_generators.cpp
)

add_executable(tests ${TEST_SOURCES})

target_link_libraries(tests
    PRIVATE
        mini_jit                 # library under test
        Catch2::Catch2WithMain   # Catch2 and its main()
)

include(Catch)
catch_discover_tests(tests)
