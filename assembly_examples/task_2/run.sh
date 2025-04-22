cd ...
[ -d build ] || mkdir build
mkdir build
cd build
cmake ../assembly_examples/task2
make
./driver
cd ../assembly_examples/task2