
/**
* @brief Identity primitive that transposes an 8x8 matrix A and return it in B.
* @param a    Pointer to column-major matrix A.
* @param b    Pointer to row-major matrix B.
* @param ld_a Leading dimension of A.
* @param ld_b Leading dimension of B.
* static void trans_neon_8_8(float const* a, float* b, int64_t ld_a, int64_t ld_b);
**/

// load A
mov x7, x0
ld1 {v0.4s-v0.3s} [x7], #16
ld1 {v4.4s-v0.7s} [x7], #16

// Transpose
// Step 1: Pairwise transpose 32-bit elements between v0/v1, v2/v3, etc.
trn1 v8.4s, v0.4s, v1.4s  
trn2 v9.4s, v0.4s, v1.4s  
trn1 v10.4s, v2.4s, v3.4s 
trn2 v11.4s, v2.4s, v3.4s 
trn1 v12.4s, v4.4s, v5.4s 
trn2 v13.4s, v4.4s, v5.4s 
trn1 v14.4s, v6.4s, v7.4s 
trn2 v15.4s, v6.4s, v7.4s   
// Step 2: Interleave 64-bit chunks
zip1 v0.2d, v8.2d, v10.2d   
zip2 v1.2d, v8.2d, v10.2d   
zip1 v2.2d, v9.2d, v11.2d   
zip2 v3.2d, v9.2d, v11.2d   
zip1 v4.2d, v12.2d, v14.2d  
zip2 v5.2d, v12.2d, v14.2d  
zip1 v6.2d, v13.2d, v15.2d  
zip2 v7.2d, v13.2d, v15.2d  

// store A
mov x8, x1
ld1 {v0.4s-v0.3s} [x8], #16
ld1 {v4.4s-v0.7s} [x8], #16