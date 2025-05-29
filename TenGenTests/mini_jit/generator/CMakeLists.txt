add_executable(BrgemmTests BrgemmTests.cpp)
target_link_libraries(BrgemmTests PRIVATE Catch2::Catch2WithMain)
add_test(NAME BrgemmTests COMMAND BrgemmTests)
