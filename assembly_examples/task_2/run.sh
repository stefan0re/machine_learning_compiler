[ -d build ] || mkdir build
cd build
gcc copy_driver.cpp copy_c.c copy_asm.s -o copy_driver
./driver
cd ..