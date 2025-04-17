Introduction to Aarch64 Assembly
================================


In the first week we want to warm up with ARM Assembly and take a closer look at the generation of machine code.
The first task deals with the generation of assembly code and its analysis. 
In the second task, we look directly at ones and zeros, e.g. our assembled add_values function.

Task 1: Hello Assembly
----------------------


In this task, a C function was given in which only `Hello Assembly Language!` is printed to the standard output.

For this function we generated assembly code once with clang and once with gcc compiler.

.. code-block:: bash
    :linenos:

    gcc -S hello_assembly.c -o hello_assembly_gcc.asm
    clang -S hello_assembly.c -o hello_assembly_clang.asm


In the resulting assembly code, the string `Hello Assembly Language!` was then marked in red.
The code that satisfies the procedure call standard was outlined in purple and the call to the libc library in green.

- Clang compiler:

.. image:: ../_static/hello_assembly_clang.png
    :alt: Clang compiler

- GCC compiler:

.. image:: ../_static/hello_assembly_gcc.png
    :alt: gcc compiler


The function was then called with the following driver:

.. code-block:: C++
    :linenos:

    #include <iostream>

    extern "C" void hello_assembly();

    using namespace std;

    int main() {
        hello_assembly();
        return 0;
    }


and the program can then be executed with the following bash script:

.. code-block:: bash
    :linenos:

    as hello_assembly_gcc.asm -o hello_assembly.o
    g++ driver.cpp hello_assembly.o -o driver
    ./driver

Our code for this task can be seen `here <https://github.com/stefan0re/machine_learning_compiler/tree/main/assembly_examples/task_1>`_.



Task 2: Assembly Function
-------------------------

For this task we use the simple function add_values that is written in Aarch64 assembly:

.. code-block::
    :linenos:

        .text
        .type add_values, %function
        .global add_values
    add_values:
        stp fp, lr, [sp, #-16]!
        mov fp, sp

        ldr w3, [x0]
        ldr w4, [x1]
        add w5, w3, w4
        str w5, [x2]

        ldp fp, lr, [sp], #16

        ret

Two 32-bit integers are loaded into the lower 32-bits of the general purpose register.
The two values are added together and the result is written back to the memory.

To assemble the :code:`add_values.s` file the following command is used:

.. code-block:: shell

        as -o add_values.o add_values.s

The :code:`-o` option specifies the output file name.

The Hexadecimal dump from the :code:`add_values.o` file can be generated with the following command:

.. code-block:: shell

        hexdump -C add_values.o > add_values_hex_dump.txt

