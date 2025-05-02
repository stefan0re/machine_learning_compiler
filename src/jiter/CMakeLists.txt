# CMakeLists.txt for jiter
file(GLOB_RECURSE JITER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/instructions/*.cpp")

add_library(jiter ${JITER_SOURCES})

target_include_directories(jiter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/instructions
)
