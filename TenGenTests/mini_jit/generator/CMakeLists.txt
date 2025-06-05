add_executable(BrgemmTests BrgemmTests.cpp)
target_link_libraries(BrgemmTests PRIVATE Catch2::Catch2WithMain TenGen)
target_include_directories(BrgemmTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME BrgemmTests COMMAND BrgemmTests)

add_executable(UnaryTests UnaryTests.cpp)
target_link_libraries(UnaryTests PRIVATE Catch2::Catch2WithMain TenGen)
target_include_directories(UnaryTests PRIVATE ${CMAKE_SOURCE_DIR}/TenGen)
add_test(NAME UnaryTests COMMAND UnaryTests)