cd ...
[ -d build ] || mkdir build
cd build
cmake ../assembly_examples/task_2
make
./driver
cd ../assembly_examples/task_2 