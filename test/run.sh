#!/bin/bash

mkdir -p build

cd build

g++ ../test_util.cpp \
    ../../src/mini_jit/generator/util.cpp \
    ../../src/mini_jit/backend/Kernel.cpp \
    ../../src/mini_jit/instructions/base.cpp \
    ../../src/mini_jit/instructions/neon.cpp \
    -o test_util

./test_util

objdump -D -b binary -m aarch64 debug_load_C.bin
objdump -D -b binary -m aarch64 debug_gen_microkernel.bin
objdump -D -b binary -m aarch64 debug_store_C.bin

cd ..