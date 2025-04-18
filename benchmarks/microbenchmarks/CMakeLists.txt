set(TARGET_NAME microbenchmark)

# Assemble

# get all benchmark files
file(GLOB ASM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/throughput/*.s")

set(ASM_OBJECTS "")
foreach(ASM_FILE IN LISTS ASM_SOURCES)
    get_filename_component(OBJ_NAME ${ASM_FILE} NAME_WE)
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_NAME}.o")

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${CMAKE_ASM_COMPILER} -o ${OBJ_FILE} ${ASM_FILE}
        DEPENDS ${ASM_FILE}
        COMMENT "Assembling ${OBJ_NAME}.s"
    )

    list(APPEND ASM_OBJECTS ${OBJ_FILE})
endforeach()

# link and compile
add_executable(${TARGET_NAME} microbenchmark_driver.cpp ${ASM_OBJECTS})