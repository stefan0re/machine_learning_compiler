#include <cstdint>
#include <cmath>
#include <iostream>
#include <chrono>

extern "C" {
    /**
      * @param a pointer to column-major matrix A.
      * @param b pointer to column-major matrix B.
      * @param c pointer to column-major matrix C.
      * @param lda leading dimension of A.
      * @param ldb leading dimension of B.
      * @param ldc leading dimension of C.
      **/
    void matmul_16_6_1( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         lda,
                        int64_t         ldb,
                        int64_t         ldc );

  /**
    * @param a pointer to column-major matrix A.
    * @param b pointer to column-major matrix B.
    * @param c pointer to column-major matrix C.
    * @param lda leading dimension of A.
    * @param ldb leading dimension of B.
    * @param ldc leading dimension of C.
    **/
void matmul_16_6_64( float   const * a,
                     float   const * b,
                     float         * c,
                     int64_t         lda,
                     int64_t         ldb,
                     int64_t         ldc );

/**
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 * @param lda leading dimension of A.
 * @param ldb leading dimension of B.
 * @param ldc leading dimension of C.
 **/
void matmul_64_6_64( float   const * a,
    float   const * b,
    float         * c,
    int64_t         lda,
    int64_t         ldb,
    int64_t         ldc );
}


void gemm_ref( float        const * i_a,
               float        const * i_b,
               float              * io_c,
               int64_t              i_m,
               int64_t              i_n,
               int64_t              i_k,
               int64_t              i_lda,
               int64_t              i_ldb,
               int64_t              i_ldc ){
                
    

                for( int l_m = 0; l_m < i_m; l_m++ ) {
                    for( int l_n = 0; l_n < i_n; l_n++ ) {
                        for( int l_k = 0; l_k < i_k; l_k++ ) {
                            io_c[ (l_n*i_ldc) + l_m ] += i_a[ (l_k*i_lda) + l_m ] * i_b[ (l_n*i_ldb) + l_k ];
                        }
                    }
                }
}

float checkDif( float * arr_1 , 
                float * arr_2 ,
                int length ){

    float result = 0.0;
    
    for(int i = 0 ; i < length ; i++){
        if( std::abs(arr_1[i] - arr_2[i]) > 0.0001){
            std::cout << "ID " << i << ": " << arr_1[i] << " / "<< arr_2[i] << std::endl;
            result = std::abs(arr_1[i] - arr_2[i]);
        }
    }  
    if( result < 1.0E-5){
        return 0.0;
    }

    return result;
}



void mm_16_6_1() {  
    
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> dur;
    uint64_t reps = 200000000;

    srand48( time(NULL) );
        
    
    int l_m = 16;
    int l_k = 1;
    int l_n = 6;
    int l_lda = l_m;
    int l_ldb = l_k;
    int l_ldc = l_m;

    double l_g_flops = 2 * l_k * l_n * l_m ;

    // initialize matrix
    float * l_a = (float *) malloc( l_k * l_m * sizeof(float));
    float * l_b = (float *) malloc( l_k * l_n * sizeof(float));
    float * l_c_1 = (float *) malloc( l_m * l_n * sizeof(float));
    float * l_c_2 = (float *) malloc( l_m * l_n * sizeof(float));

    
    for( int i = 0; i<(l_k * l_m) ; i++){
        l_a[i] = (float) drand48();
    }

    for( int i = 0; i<(l_k * l_n) ; i++){
        l_b[i] = (float)drand48() ;
    }
    

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_1[i] = (float) drand48() ;
    }

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_2[i] = l_c_1[i] ;
    }

    gemm_ref(l_a, l_b, l_c_1, l_m, l_n , l_k, l_lda, l_ldb, l_ldc );
    matmul_16_6_1(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);



    std::cout << "Error:  " << checkDif(l_c_1, l_c_2, l_m * l_n ) << std::endl;

    start = std::chrono::steady_clock::now();
    for(uint64_t i = 0 ; i < reps ; i++){
         matmul_16_6_1(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);
    }
    end = std::chrono::steady_clock::now();

    

    dur = std::chrono::duration_cast< std::chrono::duration< double> >( end - start );

    
    std::cout << "M = 16 , K = 1 , N = 6 " << std::endl;
    std::cout << "executions: " << reps << std::endl;
    std::cout << "duration: " << dur.count() << " seconds" << std::endl;
    

    l_g_flops *= reps;
    l_g_flops *= 1.0E-9;
    l_g_flops /= dur.count();

    std::cout << "GFLOPS: " << l_g_flops << std::endl;

    std::cout << "***********************" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}


void mm_16_6_64() {  
    
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> dur;
    uint64_t reps = 7000000;

    srand48( time(NULL) );
        
    
    int l_m = 16;
    int l_k = 64;
    int l_n = 6;
    int l_lda = l_m;
    int l_ldb = l_k;
    int l_ldc = l_m;

    double l_g_flops = 2 * l_k * l_n * l_m ;

    // initialize matrix
    float * l_a = (float *) malloc( l_k * l_m * sizeof(float));
    float * l_b = (float *) malloc( l_k * l_n * sizeof(float));
    float * l_c_1 = (float *) malloc( l_m * l_n * sizeof(float));
    float * l_c_2 = (float *) malloc( l_m * l_n * sizeof(float));

    
    for( int i = 0; i<(l_k * l_m) ; i++){
        l_a[i] = (float) drand48();
    }

    for( int i = 0; i<(l_k * l_n) ; i++){
        l_b[i] = (float)drand48() ;
    }
    

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_1[i] = (float) drand48() ;
    }

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_2[i] = l_c_1[i] ;
    }

    gemm_ref(l_a, l_b, l_c_1, l_m, l_n , l_k, l_lda, l_ldb, l_ldc );
    matmul_16_6_64(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);



    std::cout << "Error:  " << checkDif(l_c_1, l_c_2, l_m * l_n ) << std::endl;

    start = std::chrono::steady_clock::now();
    for(uint64_t i = 0 ; i < reps ; i++){
         matmul_16_6_64(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);
    }
    end = std::chrono::steady_clock::now();

    

    dur = std::chrono::duration_cast< std::chrono::duration< double> >( end - start );

    
    std::cout << "M = 16 , K = 64 , N = 6 " << std::endl;
    std::cout << "executions: " << reps << std::endl;
    std::cout << "duration: " << dur.count() << " seconds" << std::endl;
    

    l_g_flops *= reps;
    l_g_flops *= 1.0E-9;
    l_g_flops /= dur.count();

    std::cout << "GFLOPS: " << l_g_flops << std::endl;
    std::cout << "***********************" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}

void mm_64_6_64() {  
    
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> dur;
    uint64_t reps = 700000;

    srand48( time(NULL) );
        
    
    int l_m = 64;
    int l_k = 64;
    int l_n = 6;
    int l_lda = l_m;
    int l_ldb = l_k;
    int l_ldc = l_m;

    double l_g_flops = 2 * l_k * l_n * l_m ;

    // initialize matrix
    float * l_a = (float *) malloc( l_k * l_m * sizeof(float));
    float * l_b = (float *) malloc( l_k * l_n * sizeof(float));
    float * l_c_1 = (float *) malloc( l_m * l_n * sizeof(float));
    float * l_c_2 = (float *) malloc( l_m * l_n * sizeof(float));

    
    for( int i = 0; i<(l_k * l_m) ; i++){
        l_a[i] = (float) drand48();
    }

    for( int i = 0; i<(l_k * l_n) ; i++){
        l_b[i] = (float)drand48() ;
    }
    

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_1[i] = (float) drand48() ;
    }

    for( int i = 0; i<(l_m * l_n) ; i++){
        l_c_2[i] = l_c_1[i] ;
    }

    gemm_ref(l_a, l_b, l_c_1, l_m, l_n , l_k, l_lda, l_ldb, l_ldc );
    matmul_64_6_64(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);



    std::cout << "Error:  " << checkDif(l_c_1, l_c_2, l_m * l_n ) << std::endl;

    start = std::chrono::steady_clock::now();
    for(uint64_t i = 0 ; i < reps ; i++){
         matmul_64_6_64(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);
    }
    end = std::chrono::steady_clock::now();

    

    dur = std::chrono::duration_cast< std::chrono::duration< double> >( end - start );

    
    std::cout << "M = 64 , K = 64 , N = 6 " << std::endl;
    std::cout << "executions: " << reps << std::endl;
    std::cout << "duration: " << dur.count() << " seconds" << std::endl;
    

    l_g_flops *= reps;
    l_g_flops *= 1.0E-9;
    l_g_flops /= dur.count();

    std::cout << "GFLOPS: " << l_g_flops << std::endl;
    std::cout << "***********************" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}


int main(){
    mm_16_6_1();
    mm_16_6_64();
    mm_64_6_64();

    return EXIT_SUCCESS;
}
