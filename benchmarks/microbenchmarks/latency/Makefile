BUILD_DIR = ./build


all: driver.cpp kernels/add.s kernels/mul.s 
		gcc -pedantic -Wall -Wextra -Werror -c kernels/add.s -o ${BUILD_DIR}/add.o
		gcc -pedantic -Wall -Wextra -Werror -c kernels/mul.s -o ${BUILD_DIR}/mul.o
		g++ -pedantic -Wall -Wextra -Werror -O2 driver.cpp ${BUILD_DIR}/add.o ${BUILD_DIR}/mul.o -o ${BUILD_DIR}/bench


$(shell mkdir -p ${BUILD_DIR})