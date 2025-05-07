[ -d build ] || mkdir build
cd build
g++ ../test_driver.cpp ../instructions.h ../base.cpp -o test_driver
./test_driver
cd ..