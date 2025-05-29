add_executable(BrgemmTests BrgemmTests.cpp)
target_link_libraries(BrgemmTests PRIVATE Catch2::Catch2WithMain TenGen)
target_include_directories(BrgemmTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME BrgemmTests COMMAND BrgemmTests)