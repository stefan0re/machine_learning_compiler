#!/bin/bash

mkdir -p build

cd build

g++ ../test_gen_methods.cpp \
    ../../src/mini_jit/generator/Util.cpp \
    ../../src/mini_jit/backend/Kernel.cpp \
    ../../src/mini_jit/instructions/base.cpp \
    ../../src/mini_jit/instructions/neon.cpp \
    ../../src/mini_jit/generator/Unary.cpp \
    ../test_utils.cpp \
    -o test_gen_methods

./test_gen_methods

objdump -D -b binary -m aarch64 output_test.bin

cd ..