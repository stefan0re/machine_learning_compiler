set(TARGET_NAME driver_copy)

set(SOURCES
    copy_c.c
    copy_asm.s
    copy_driver.cpp
)

# Create the executable
add_executable(${TARGET_NAME} ${SOURCES})