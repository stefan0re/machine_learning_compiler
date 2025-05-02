set(TARGET_NAME copy_driver)

set(SRC_CPP copy_driver.cpp)
set(SRC_C copy_c.c)
set(SRC_ASM ${CMAKE_CURRENT_SOURCE_DIR}/copy_asm.s)

get_filename_component(ASM_OBJ_NAME ${SRC_ASM} NAME_WE)
set(ASM_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ASM_OBJ_NAME}.o)

add_custom_command(
    OUTPUT ${ASM_OBJ}
    COMMAND ${CMAKE_ASM_COMPILER} -o ${ASM_OBJ} ${SRC_ASM}
    DEPENDS ${SRC_ASM}
)

add_custom_target(copy_asm_obj DEPENDS ${ASM_OBJ})

add_executable(${TARGET_NAME} ${SRC_CPP} ${SRC_C} ${ASM_OBJ})

add_dependencies(${TARGET_NAME} copy_asm_obj)