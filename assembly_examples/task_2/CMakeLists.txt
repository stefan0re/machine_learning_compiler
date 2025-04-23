cmake_minimum_required(VERSION 3.10)
project(driver_copy C CXX ASM)

set(SOURCES
    copy_c.c
    copy_asm.s
    copy_driver.cpp
)

# Create the executable
add_executable(copy_driver ${SOURCES})