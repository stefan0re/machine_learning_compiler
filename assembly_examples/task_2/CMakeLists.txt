cmake_minimum_required(VERSION 3.10)
project(task2 C CXX ASM)

set(SOURCES
    copy_c.c
    copy_asm.s
    copy_driver.cpp
)

# Create the executable
add_executable(copy_driver ${SOURCES})