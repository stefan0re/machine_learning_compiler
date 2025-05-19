#!/bin/bash

mkdir -p build

cd build

as ../kernels/trans_neon_8_8.s -o trans_neon_8_8.o
as ../kernels/matmul_1.s -o matmul_1.o
as ../kernels/matmul_2.s -o matmul_2.o
as ../kernels/matmul_64_64_64.s -o matmul_64_64_64.o

g++ ../driver.cpp trans_neon_8_8.o matmul_1.o matmul_2.o matmul_64_64_64.o -o driver

./driver

cd ..