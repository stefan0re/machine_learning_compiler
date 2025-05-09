# Base-Instruction-Tests
add_executable(mini_jit_base_instr_tests
    test_base_instructions.cpp
)
target_link_libraries(mini_jit_base_instr_tests PRIVATE mini_jit)
add_test(NAME BaseInstrTests COMMAND mini_jit_base_instr_tests)

# Neon-Instruction-Tests
add_executable(mini_jit_neon_instr_tests
    test_neon_instructions.cpp
)
target_link_libraries(mini_jit_neon_instr_tests PRIVATE mini_jit)
add_test(NAME NeonInstrTests COMMAND mini_jit_neon_instr_tests)

# Mini-Jit-Tests
add_executable(mini_jit_unit_tests
    test_mini_jit.cpp
)
target_link_libraries(mini_jit_unit_tests PRIVATE mini_jit)
add_test(NAME MiniJitTests COMMAND mini_jit_unit_tests)
