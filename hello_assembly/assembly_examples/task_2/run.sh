[ -d build ] || mkdir build
cd build
gcc -c ../copy_c.c -o copy_c.o
g++ ../copy_driver.cpp copy_c.o ../copy_asm.s -o copy_driver
./copy_driver
cd ..