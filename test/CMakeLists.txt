# Base-Instruction-Tests
add_executable(mini_jit_base_instr_tests
    test_base_instructions.cpp
    test_utils.cpp
)
target_link_libraries(mini_jit_base_instr_tests PRIVATE mini_jit)
add_test(NAME BaseInstrTests COMMAND mini_jit_base_instr_tests)

# Neon-Instruction-Tests
add_executable(mini_jit_neon_instr_tests
    test_neon_instructions.cpp
    test_utils.cpp
)
target_link_libraries(mini_jit_neon_instr_tests PRIVATE mini_jit)
add_test(NAME NeonInstrTests COMMAND mini_jit_neon_instr_tests)

# Generator-Methods-Tests
add_executable(mini_jit_generator_methods_test
    test_gen_methods.cpp
    test_utils.cpp
)
target_link_libraries(mini_jit_generator_methods_test PRIVATE mini_jit)
add_test(NAME GeneratorMethodsTests COMMAND mini_jit_generator_methods_test)

