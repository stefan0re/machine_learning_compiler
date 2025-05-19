#!/bin/bash

mkdir -p build

cd build

g++ ../driver.cpp \
    ../kernsels/trans_neon_8_8.s \
    -o driver

./driver

cd ..