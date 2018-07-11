
#include <iostream>
#include <cstdlib>
#include "microtime.h"
#include <stdio.h>


#define D 2048

#ifdef _MKL_
#include <mkl_cblas.h>
#include <mkl.h>
#elif _OPENBLAS_
#include <cblas.h>
#endif



int main(int argc, char** argv){




#ifdef _MKL_
	const CBLAS_LAYOUT Order=CblasRowMajor;
	const CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const CBLAS_TRANSPOSE TransB=CblasNoTrans;
#elif _OPENBLAS_
	const enum CBLAS_ORDER Order=CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
#endif
	const int M= D;//numRows of A; numCols of C
	const int N= D;//numCols of B and C
	const int K= D;//numCols of A; numRows of B
	const float alpha=1;
	const float beta=1;
	const int lda=K;//numCols of A
	const int ldb=N;//numCols of B
	const int ldc=N;//numCols of C

#ifdef _MKL_
    mkl_set_num_threads(1);
#endif

    double* A = (double*) _mm_malloc ( sizeof(double) * D * (D+1), 64);
    double* B = (double*) _mm_malloc ( sizeof(double) * D * D, 64);
    double* C = (double*) _mm_malloc ( sizeof(double) * D * D, 64);

//    srand(292);

    for(int i=0; i< D*D; i++)
    {
        
        A[i] = (double(i)/3);
        B[i] = (double(i)/7);
        C[i] = (double(i)/11);

    }

for(int i=0; i<10; i++)
{
    double t1 = microtime();
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
    
    double t2 = microtime();
    std::cout<<" C[0] = "<< C[0]<<std::endl;

    std::cout<<" elapsed time when D="<< D<<" is "<< t2 - t1 << "s"<<std::endl;


}
}
