[ -d build ] || mkdir build
cd build
cmake ..
make
./driver
cd ..