## Copying Data
The data directory contains the functions copy_c_0 and copy_c_1 in the file copy_c.c. The function copy_c_0 simply copies seven int32_t values from one array to another. The copy_c_1 function has an additional input parameter n specifying the number of values to be copied.

Additionally, a driver is provided in the file copy_driver.cpp. The driver calls copy_c_0 and copy_c_1 on sample data and checks the results. The driver also calls the unfinished functions copy_asm_0 and copy_asm_1, which are located in the file copy_asm.s.

Tasks

1. Implement the function copy_asm_0 using only base instructions. The function should have the same functionality as its C counterpart.
2. Implement the function copy_asm_1 using only base instructions. The function should have the same functionality as its C counterpart.
