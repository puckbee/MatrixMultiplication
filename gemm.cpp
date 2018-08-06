
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "microtime.h"
#include <stdio.h>
#include <omp.h>

#include <immintrin.h>

#include "tile.h"

#ifdef _MKL_
#include <mkl_cblas.h>
#include <mkl.h>
#elif _OPENBLAS_
#include <cblas.h>
#endif

#ifndef D
#define D 1024
#endif

#ifndef S 
#define S 4
#endif

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc );

void MKL_MMult( int m, int n, int k, double *a, int lda,
                                     double *b, int ldb,
                                     double *c, int ldc);


void MKL_MMult( int m, int n, int k, double *a, int lda,
                                     double *b, int ldb,
                                     double *c, int ldc)
{

    double* tmpC = (double*) _mm_malloc ( sizeof(double) * D * D, 64);
    memcpy(tmpC,c, sizeof(double)*D*D);

#ifdef _MKL_
	const CBLAS_LAYOUT Order=CblasRowMajor;
	const CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const CBLAS_TRANSPOSE TransB=CblasNoTrans;
#elif _OPENBLAS_
	const enum CBLAS_ORDER Order=CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
#endif
	const int M= m;//numRows of A; numCols of C
	const int N= n;//numCols of B and C
	const int K= k;//numCols of A; numRows of B
	const float alpha=1;
	const float beta=1;

#ifdef _MKL_
//    mkl_set_num_threads(1);
#endif
    
    
    double t1 = microtime();
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, a, lda, b,
      ldb, beta, c, N);
    double t2 = microtime();
    std::cout<<" Standard: elapsed time when D="<< D<<" is "<< t2 - t1 << "s"<<std::endl;


    for(int i=0; i<10; i++)
    {
        double t1 = microtime();
        cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, a, lda, b,
          ldb, beta, tmpC, N);
        
        double t2 = microtime();
        std::cout<<" c[0] = "<< tmpC[0]<<std::endl;

        std::cout<<" elapsed time when D="<< D<<" is "<< t2 - t1 << "s"<<std::endl;
    }
}

int main(int argc, char** argv)
{

    double* A = (double*) _mm_malloc ( sizeof(double) * D * (D+1), 64);
    double* B = (double*) _mm_malloc ( sizeof(double) * D * D, 64);
    double* C = (double*) _mm_malloc ( sizeof(double) * D * D, 64);
    double* C2 = (double*) _mm_malloc ( sizeof(double) * D * D, 64);
    double* refC = (double*) _mm_malloc ( sizeof(double) * D * D, 64);


//    srand(292);

    for(int i=0; i< D*D; i++)
    {
        
//        A[i] = (double(i%100)/2) * (double(i%100)/2);
//        B[i] = (double(i%100)/4) * (double(i%100)/4);
        A[i] = i/2;
        B[i] = i/2;
//        C[i] = (double(i%100)/2);
        C[i] = 0;
        

    }
    
    memcpy(C2,C, sizeof(double)*D*D);
        
    double t1 = microtime();
    MY_MMult(D, D, D, A, D, B, D, C, D);
    double t2 = microtime();
    std::cout<<" Standard: elapsed time when D="<< D<<" is "<< t2 - t1 << "s"<<std::endl;


    for(int i=0; i<10; i++)
    {
        double t1 = microtime();
        MY_MMult(D, D, D, A, D, B, D, C2, D);
        
        double t2 = microtime();
        std::cout<<" C[0] = "<< C2[0]<<std::endl;

        std::cout<<" elapsed time when D="<< D<<" is "<< t2 - t1 << "s"<<std::endl;

    }

    

    MKL_MMult(D, D, D, A, D, B, D, refC, D);


    int err=0;
    for(int i=0; i < D*D; i++)
    {
       if(std::abs(refC[i] - C[i]) > 0)
       {
           err++;
           printf(" refC[%d] = %f; C[%d] = %f \n", i, refC[i], i, C[i]);
       }
//       std::cout<<" refC["<<i<<"] = "<<refC[i]<<"; C["<<i<<"] = "<<C[i]<<std::endl;
    }

    if(err==0)
        std::cout<<" Check Pass! "<<std::endl;
    else
        std::cout<<" "<<err<<" errors occurred"<<std::endl;

}

/* Block sizes */
#define kc 128
#define nc 512
#define mc 4096
#define mcc 96
#define ncc 128

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot6x8( int, double *, int, double *, int, double *, int, double*, int, int);
void AddDot4x8( int, double *, int, double *, int, double *, int, double*, int, int);
void AddDot2x8( int, double *, int, double *, int, double *, int, double*, int, int);
void PackB_and_AddDot6x8( int, double *, int, double *, int, double *, int, double *, int, double*, int, int, int, int);
void PackA_and_AddDot6x8( int k, double *oa, int lda, double *a,  double *b, int ldb, double *c, int ldc, double* packedC, int firstKC, int lastKC);
void PackMatrixA( int, double *, int, double *, int, int);
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, int, double *, int, int, double*, double*, double*, int, int, int, int, int);
void InnerKernel12x4( int, int, int, double *, int, double *, int, double *, int, int, double*, double*);
void OutterKernel( int, int, int, double *, int, double *, int, double *, int, int, double*, double*, double*, int, int, int);
void PackKernel( int, int, int, double *, int, double *, int, double *, int, int, double*, double*);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int s, sb;

  double* packedA = (double*) _mm_malloc(sizeof(double) * kc * mcc * 4, 64);        // occupied a small area of memory
//  double* packedA2 = (double*) _mm_malloc(sizeof(double) * kc * mc * 4, 64);
  double* packedA2 = NULL;
  double* packedB = (double*) _mm_malloc(sizeof(double) * kc * ncc * 4, 64);
  double* packedC = (double*) _mm_malloc(sizeof(double) * nc * mc * 4, 64);


//  memset(packedC, 0, sizeof(double) * nc * m);
//  double* packedB = (double*) _mm_malloc(sizeof(double) * D * D * 2, 64);

  int Nthrds = omp_get_num_threads();
  printf("Nthrds = %d \n", Nthrds);


      for(s=0; s<m; s+=mc){
          sb = min(m-s, mc);

#pragma omp parallel num_threads(4)
// for(int idx=0; idx<4;idx++)
 {
     int idx = omp_get_thread_num();
     int i, p, pb, ib, si, sib, sj, ssj;
/*
      for (p=0; p<k; p+=kc ){
         pb = min( k-p, kc );
         for(si=0; si<sb; si+=mcc)
         {
           sib = min( sb-si, mcc );
           ssj=0;
           for ( ssj=0; ssj<(sib/6*6); ssj+=6 ){       
               printf(" A [ %d, %d ]\n", s+si+ssj, p);
                PackMatrixA( pb, &A( s + si+ssj, p ), lda, &packedA2[kc*mc*idx+ (s+si+ssj)*kc ], ssj, sib);
                printf(" packedA2[0] = %f\n", packedA2[0]);

           }
         }
      }
*/


    for (i=idx*(n/nc/4)*nc; i<(idx+1)*(n/nc/4)*nc; i+=nc ){
      ib = min( n-i, nc );
      for (p=0; p<k; p+=kc ){
      pb = min( k-p, kc );
//      InnerKernel( m, ib, pb, &A( 0,p ), lda, &B(p, i ), ldb, &C( 0,i ), ldc, i==0, packedA, packedB);
//      OutterKernel( m, ib, pb, &A( 0,p ), lda, &B(p, i ), ldb, &C( 0,i ), ldc, i==0, packedA, packedB + p*n+i*nc);
      OutterKernel( sb, ib, pb, &A( s,p ), lda, &B(p, i ), ldb, &C( s,i ), ldc, i==0, packedA + kc*mcc*idx, packedB + kc*ncc*idx, packedC + nc*mc*idx, p==0, (p+pb)>=k, idx);
    }
    }
 }
  }

}

void OutterKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time, double* packedA, double* packedB, double* packedC, int firstKC, int lastKC, int idx)
{
    int i,j,p, ib, jb;

    for(i=0; i<m; i+=mcc)
    {
      ib = min( m-i, mcc );
        for(j=0; j<n; j+=ncc)
        {
//            jb = min(n-j,ncc);
//           if(i == 2016 && j == 352)
//           printf(" n = %d, mcc = %d, ncc = %d, firstKC = %d, packedC[0] = %f \n", n, mcc, ncc, firstKC, packedC[i*n+j*mcc]);
           InnerKernel( ib, ncc, k, &A( i,0 ), lda, &B(0,j ), ldb, &C( i,j ), ldc, j==0, packedA, packedB, packedC + i * n + j * ib, firstKC, lastKC, i,j, idx); // (i/mcc * n/ncc + j/ncc) * mcc * ncc;
        }
    }
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time, double* packedA, double* packedB, double* packedC, int firstKC, int lastKC, int outi, int outj, int idx)
{
  int i, j, sp;
//         if(outi ==2016 && outj == 352)
//           printf(" when enter the function: firstKC = %d, packedC[0] = %f \n", firstKC, packedC[0]);

  for ( j=0; j<(m/6*6); j+=6 ){        /* Loop over the columns of C, unrolled by 4 */
    if ( first_time )
    {
      PackMatrixA( k, &A( j, 0 ), lda, &packedA[ j*k ], j, m);
/*
      for(sp=j*k; sp < (j+1)*k; sp++)
      {
          int err=0;
         if(std::abs(packedA[i] - packedA2[i]) > 0)
         {
            err++;
            printf(" packedA[%d] = %f; packedA2[%d] = %f \n", sp, packedA[sp], sp, packedA2[sp]);
         }
         if(err == 0)
             printf(" check packedA pass!\n");

       }
*/       
    }
    for ( i=0; i<n; i+=8 ){        /* Loop over the rows of C */
      if ( j == 0 )
      {
//	      PackMatrixB( k, &B( 0, i ), ldb, &packedB[ i*k ]);

//         if(outi ==2016 && outj == 352 && !firstKC)
//           printf(" in Inner kernel packedC[0] = %f \n", packedC[0]);
          PackB_and_AddDot6x8( k, &packedA[ j*k ], 6, &B( 0, i ), ldb, &packedB[ i*k ], 8, &C( j,i ), ldc, packedC + i*6, firstKC, lastKC, outi, outj);
//          AddDot6x8( k, &packedA[ j*k ], lda, &packedB[ i*k ], ldb, &C( j,i ), ldc, packedC + j*n + i*6, firstKC, lastKC);
//          if(outi == 2016 && outj == 352 && !firstKC)
//              printf("outi = %d, outj = %d, c[0] = %f, packedC[0] = %f \n", outi, outj, C(j,i), packedC+j*n+i*6);
      }
//      else if(j==60)
//          AddDot4x8( k, &packedA[ j*k ], lda, &packedB[ i*k ], ldb, &C( j,i ), ldc, packedC + j*n + i*4, firstKC, lastKC);
      else
      {
            AddDot6x8( k, &packedA[ j*k ], lda, &packedB[ i*k ], ldb, &C( j,i ), ldc, packedC + j*n + i*6, firstKC, lastKC);
      }
    }
  }

  if(m==32)
  {
  for ( j=30; j<m; j+=2 ){       // Loop over the columns of C, unrolled by 4 
    if ( first_time )
    {
      PackMatrixA( k, &A( j, 0 ), lda, &packedA[ j*k ], j, m);
    }
    for ( i=0; i<n; i+=8 ){        // Loop over the rows of C 
       AddDot2x8( k, &packedA[ j*k ], 6, &packedB[ i*k ], 8, &C( j,i ), ldc, packedC + j*n + i*2, firstKC, lastKC);
    }
  }
  }

  if(m==64)
  {
  for ( j=60; j<m; j+=4 ){       // Loop over the columns of C, unrolled by 4 
    if ( first_time )
    {
      PackMatrixA( k, &A( j, 0 ), lda, &packedA[ j*k ], j, m);
    }
    for ( i=0; i<n; i+=8 ){        // Loop over the rows of C 
       AddDot4x8( k, &packedA[ j*k ], 6, &packedB[ i*k ], 8, &C( j,i ), ldc, packedC + j*n + i*4, firstKC, lastKC);
    }
  }
  }

}

void PackMatrixB( int k, double *b, int ldb, double *b_to)
{
  int j;

  __m256d reg1;

#pragma unroll(4)
//#pragma noprefetch
  for( j=0; j<k; j++){  /* loop over columns of A */
    double 
        *b_ij_pntr = b + j * ldb;

    reg1 = _mm256_load_pd(b_ij_pntr);
    _mm256_store_pd(b_to, reg1);
    b_to += 4;

    reg1 = _mm256_load_pd(b_ij_pntr + 4);
    _mm256_store_pd(b_to, reg1);
    b_to += 4;

  }
}


void PackMatrixA( int k, double *a, int lda, double *a_to, int j, int m)
{
  int i;

  if(m==64 && j==60)       // only when 6x8; mcc = 64; in the last pack iteration
  {
      double 
        *a_i0_pntr = &A( 0, 0 ), *a_i1_pntr = &A( 1, 0 ),
        *a_i2_pntr = &A( 2, 0 ), *a_i3_pntr = &A( 3, 0 );
      
      v4df_t vreg;
#pragma unroll(4)
      for( i=0; i<k; i++){ //  loop over rows of B 
        *a_to++ = *a_i0_pntr++;
        *a_to++ = *a_i1_pntr++;
        *a_to++ = *a_i2_pntr++;
        *a_to++ = *a_i3_pntr++;
      }

  }
  else if (m==32 &&j==30)
  {
      double 
        *a_i0_pntr = &A( 0, 0 ), *a_i1_pntr = &A( 1, 0 );
      
      v4df_t vreg;
#pragma unroll(4)
      for( i=0; i<k; i++){ //  loop over rows of B 
        *a_to++ = *a_i0_pntr++;
        *a_to++ = *a_i1_pntr++;
      }


  }
  else
      
  {
      double 
        *a_i0_pntr = &A( 0, 0 ), *a_i1_pntr = &A( 1, 0 ),
        *a_i2_pntr = &A( 2, 0 ), *a_i3_pntr = &A( 3, 0 ),
        *a_i4_pntr = &A( 4, 0 ), *a_i5_pntr = &A( 5, 0 );

      v4df_t vreg;


#pragma unroll(4)
      for( i=0; i<k; i++){  /* loop over rows of B */

        *a_to++ = *a_i0_pntr++;
//        printf(" move %f from packedA to packedA2\n", *(a_to-1));
        *a_to++ = *a_i1_pntr++;
//        printf(" move %f from packedA to packedA2\n", *(a_to-1));
        *a_to++ = *a_i2_pntr++;
        *a_to++ = *a_i3_pntr++;
        *a_to++ = *a_i4_pntr++;
        *a_to++ = *a_i5_pntr++;
        
      }
  }

}

int print256(std::string msg, v4df_t x)
{

//    printf(" msg: %s \n", msg);
    std::cout<<" msg: "<< msg <<std::endl;
    for(int i=0; i< 4; i++)
        printf(" %f ", x.d[i]);
    printf(" \n");

    return 0;

}

void AddDot6x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc, double* packedC, int firstKC, int lastKC)
{

  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c04_vreg,
      c10_vreg, c14_vreg,
      c20_vreg, c24_vreg,
      c30_vreg, c34_vreg,
      c40_vreg, c44_vreg,
      c50_vreg, c54_vreg,

    b00_vreg,
    b04_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);

  c00_vreg.v = zero_vreg.v; 
  c04_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c14_vreg.v = zero_vreg.v;
  c20_vreg.v = zero_vreg.v;
  c24_vreg.v = zero_vreg.v;
  c30_vreg.v = zero_vreg.v;
  c34_vreg.v = zero_vreg.v;
  c40_vreg.v = zero_vreg.v;
  c44_vreg.v = zero_vreg.v;
  c50_vreg.v = zero_vreg.v;
  c54_vreg.v = zero_vreg.v;

  __m128d ro2;

#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    
    _mm_prefetch((const char *)(b+32), _MM_HINT_T0);
    _mm_prefetch((const char *)(b+36), _MM_HINT_T0);
    _mm_prefetch((const char *)(a+24), _MM_HINT_T0);
    
    b00_vreg.v = _mm256_load_pd( (double *) b );
    b04_vreg.v = _mm256_load_pd( (double *) (b+4) );


    /* First row and second rows */

//    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
//    a0_vreg.v = _mm256_load_pd( (double *) a );       /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);
    c04_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c04_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);
    c14_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c14_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
//    a0_vreg.v = _mm256_permute2f128_pd(a0_vreg.v, a0_vreg.v, 0b00000001);
//    a0_vreg.v = _mm256_permute4x64_pd(a0_vreg.v, 0b01001110);
    ro2 = _mm_load_pd((double*)(a+2));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);
    c24_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c24_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);
    c34_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c34_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+4) );   /* load and duplicate */

    ro2 = _mm_load_pd((double*)(a+4));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c40_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c40_vreg.v);
    c44_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c44_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+5) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c50_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c50_vreg.v);
    c54_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c54_vreg.v);

    a += 6;
    b += 8;

  }

/*
      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 
      c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
      c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(&C(4,4))); 
      c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
      c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(&C(5,4)));

      _mm256_store_pd(&C(0, 0), c00_vreg.v);
      _mm256_store_pd(&C(0, 4), c04_vreg.v);
      _mm256_store_pd(&C(1, 0), c10_vreg.v);
      _mm256_store_pd(&C(1, 4), c14_vreg.v);
      _mm256_store_pd(&C(2, 0), c20_vreg.v);
      _mm256_store_pd(&C(2, 4), c24_vreg.v);
      _mm256_store_pd(&C(3, 0), c30_vreg.v);
      _mm256_store_pd(&C(3, 4), c34_vreg.v);
      _mm256_store_pd(&C(4, 0), c40_vreg.v);
      _mm256_store_pd(&C(4, 4), c44_vreg.v);
      _mm256_store_pd(&C(5, 0), c50_vreg.v);
      _mm256_store_pd(&C(5, 4), c54_vreg.v);
*/



    if(lastKC)
    {  
      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 
      c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(packedC+32));
      c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(packedC+36)); 
      c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(packedC+40));
      c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(packedC+44));

      b00_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b0101);
      c00_vreg.v = b00_vreg.v;
      c10_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b0101);
      c04_vreg.v = b00_vreg.v;
      c14_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b0101);
      c20_vreg.v = b00_vreg.v;
      c30_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b0101);
      c24_vreg.v = b00_vreg.v;
      c34_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c40_vreg.v, c50_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c40_vreg.v, c50_vreg.v, 0b0101);
      c40_vreg.v = b00_vreg.v;
      c50_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c44_vreg.v, c54_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c44_vreg.v, c54_vreg.v, 0b0101);
      c44_vreg.v = b00_vreg.v;
      c54_vreg.v = b04_vreg.v;


      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 
      c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
      c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(&C(4,4))); 
      c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
      c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(&C(5,4)));

      _mm256_store_pd(&C(0, 0), c00_vreg.v);
      _mm256_store_pd(&C(0, 4), c04_vreg.v);
      _mm256_store_pd(&C(1, 0), c10_vreg.v);
      _mm256_store_pd(&C(1, 4), c14_vreg.v);
      _mm256_store_pd(&C(2, 0), c20_vreg.v);
      _mm256_store_pd(&C(2, 4), c24_vreg.v);
      _mm256_store_pd(&C(3, 0), c30_vreg.v);
      _mm256_store_pd(&C(3, 4), c34_vreg.v);
      _mm256_store_pd(&C(4, 0), c40_vreg.v);
      _mm256_store_pd(&C(4, 4), c44_vreg.v);
      _mm256_store_pd(&C(5, 0), c50_vreg.v);
      _mm256_store_pd(&C(5, 4), c54_vreg.v);
    }
    else
    
    {
        if(!firstKC)
        {
          c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
          c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
          c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
          c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
          c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
          c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
          c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
          c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 
          c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(packedC+32));
          c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(packedC+36)); 
          c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(packedC+40));
          c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(packedC+44));
        }

      _mm256_store_pd(packedC+0, c00_vreg.v);
      _mm256_store_pd(packedC+4, c04_vreg.v);
      _mm256_store_pd(packedC+8, c10_vreg.v);
      _mm256_store_pd(packedC+12, c14_vreg.v);
      _mm256_store_pd(packedC+16, c20_vreg.v);
      _mm256_store_pd(packedC+20, c24_vreg.v);
      _mm256_store_pd(packedC+24, c30_vreg.v);
      _mm256_store_pd(packedC+28, c34_vreg.v);
      _mm256_store_pd(packedC+32, c40_vreg.v);
      _mm256_store_pd(packedC+36, c44_vreg.v);
      _mm256_store_pd(packedC+40, c50_vreg.v);
      _mm256_store_pd(packedC+44, c54_vreg.v);
    }

/*
   c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(0,8))); 
  c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(0,12)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(0,16))); 
  c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(0,20))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(0,24)));
  c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(0,28))); 
  c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(0,32)));
  c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(&C(0,36))); 
  c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(0,40)));
  c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(&C(0,44)));

  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(0, 4), c04_vreg.v);
  _mm256_store_pd(&C(0, 8), c10_vreg.v);
  _mm256_store_pd(&C(0, 12), c14_vreg.v);
  _mm256_store_pd(&C(0, 16), c20_vreg.v);
  _mm256_store_pd(&C(0, 20), c24_vreg.v);
  _mm256_store_pd(&C(0, 24), c30_vreg.v);
  _mm256_store_pd(&C(0, 28), c34_vreg.v);
  _mm256_store_pd(&C(0, 32), c40_vreg.v);
  _mm256_store_pd(&C(0, 36), c44_vreg.v);
  _mm256_store_pd(&C(0, 40), c50_vreg.v);
  _mm256_store_pd(&C(0, 44), c54_vreg.v);
*/
}


void AddDot4x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc, double* packedC, int firstKC, int lastKC)
{

  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c04_vreg,
      c10_vreg, c14_vreg,
      c20_vreg, c24_vreg,
      c30_vreg, c34_vreg,

    b00_vreg,
    b04_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);

  c00_vreg.v = zero_vreg.v; 
  c04_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c14_vreg.v = zero_vreg.v;
  c20_vreg.v = zero_vreg.v;
  c24_vreg.v = zero_vreg.v;
  c30_vreg.v = zero_vreg.v;
  c34_vreg.v = zero_vreg.v;

  __m128d ro2;

#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    
    b00_vreg.v = _mm256_load_pd( (double *) b );
    b04_vreg.v = _mm256_load_pd( (double *) (b+4) );
    b += 8;


    /* First row and second rows */

//    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
//    a0_vreg.v = _mm256_load_pd( (double *) a );       /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);
    c04_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c04_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);
    c14_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c14_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a + 2));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);
    c24_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c24_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);
    c34_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c34_vreg.v);

    a += 4;

/*
  if(profiling)
  {
     print256(" in computing a ", a_0p_a_1p_a_2p_a_3p_vreg);
     print256(" in computing b ", b_p0_vreg);
     print256(" in computing c ", c_00_c_10_c_20_c_30_vreg);
  }
*/

  }

  /*
  
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
  c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(0, 4), c04_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(1, 4), c14_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(2, 4), c24_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);
  _mm256_store_pd(&C(3, 4), c34_vreg.v);

*/


    if(lastKC)
    {  
      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 

      b00_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b0101);
      c00_vreg.v = b00_vreg.v;
      c10_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b0101);
      c04_vreg.v = b00_vreg.v;
      c14_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b0101);
      c20_vreg.v = b00_vreg.v;
      c30_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b0101);
      c24_vreg.v = b00_vreg.v;
      c34_vreg.v = b04_vreg.v;



      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 

      _mm256_store_pd(&C(0, 0), c00_vreg.v);
      _mm256_store_pd(&C(0, 4), c04_vreg.v);
      _mm256_store_pd(&C(1, 0), c10_vreg.v);
      _mm256_store_pd(&C(1, 4), c14_vreg.v);
      _mm256_store_pd(&C(2, 0), c20_vreg.v);
      _mm256_store_pd(&C(2, 4), c24_vreg.v);
      _mm256_store_pd(&C(3, 0), c30_vreg.v);
      _mm256_store_pd(&C(3, 4), c34_vreg.v);
    }
    else
    
    {
        if(!firstKC)
        {
          c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
          c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
          c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
          c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
          c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
          c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
          c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
          c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 
        }

      _mm256_store_pd(packedC+0, c00_vreg.v);
      _mm256_store_pd(packedC+4, c04_vreg.v);
      _mm256_store_pd(packedC+8, c10_vreg.v);
      _mm256_store_pd(packedC+12, c14_vreg.v);
      _mm256_store_pd(packedC+16, c20_vreg.v);
      _mm256_store_pd(packedC+20, c24_vreg.v);
      _mm256_store_pd(packedC+24, c30_vreg.v);
      _mm256_store_pd(packedC+28, c34_vreg.v);
    }

}



void AddDot2x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc, double* packedC, int firstKC, int lastKC)
{

  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c04_vreg,
      c10_vreg, c14_vreg,

    b00_vreg,
    b04_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);

  c00_vreg.v = zero_vreg.v; 
  c04_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c14_vreg.v = zero_vreg.v;

  __m128d ro2;

#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    
    b00_vreg.v = _mm256_load_pd( (double *) b );
    b04_vreg.v = _mm256_load_pd( (double *) (b+4) );
    b += 8;


    /* First row and second rows */

//    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
//    a0_vreg.v = _mm256_load_pd( (double *) a );       /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);
    c04_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c04_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);
    c14_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c14_vreg.v);

    a += 2;

/*
  if(profiling)
  {
     print256(" in computing a ", a_0p_a_1p_a_2p_a_3p_vreg);
     print256(" in computing b ", b_p0_vreg);
     print256(" in computing c ", c_00_c_10_c_20_c_30_vreg);
  }
*/

  }

  /*
  
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
  c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(0, 4), c04_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(1, 4), c14_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(2, 4), c24_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);
  _mm256_store_pd(&C(3, 4), c34_vreg.v);

*/


    if(lastKC)
    {  
      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   

      b00_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b0101);
      c00_vreg.v = b00_vreg.v;
      c10_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b0101);
      c04_vreg.v = b00_vreg.v;
      c14_vreg.v = b04_vreg.v;



      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   

      _mm256_store_pd(&C(0, 0), c00_vreg.v);
      _mm256_store_pd(&C(0, 4), c04_vreg.v);
      _mm256_store_pd(&C(1, 0), c10_vreg.v);
      _mm256_store_pd(&C(1, 4), c14_vreg.v);
    }
    else
    
    {
        if(!firstKC)
        {
          c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
          c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
          c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
          c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
        }

      _mm256_store_pd(packedC+0, c00_vreg.v);
      _mm256_store_pd(packedC+4, c04_vreg.v);
      _mm256_store_pd(packedC+8, c10_vreg.v);
      _mm256_store_pd(packedC+12, c14_vreg.v);
    }

}





void PackB_and_AddDot6x8( int k, double *a, int lda, double *ob, int ldb,  double *b, int ldb2, double *c, int ldc, double* packedC, int firstKC, int lastKC, int outi, int outj)
{
  int j;

  double* b_to = b;
  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c04_vreg,
      c10_vreg, c14_vreg,
      c20_vreg, c24_vreg,
      c30_vreg, c34_vreg,
      c40_vreg, c44_vreg,
      c50_vreg, c54_vreg,

    b00_vreg,
    b04_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);
//  zero_vreg.v = _mm256_set_pd(1.0,2.0,3.0,4.0);

  c00_vreg.v = zero_vreg.v; 
  c04_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c14_vreg.v = zero_vreg.v;
  c20_vreg.v = zero_vreg.v;
  c24_vreg.v = zero_vreg.v;
  c30_vreg.v = zero_vreg.v;
  c34_vreg.v = zero_vreg.v;
  c40_vreg.v = zero_vreg.v;
  c44_vreg.v = zero_vreg.v;
  c50_vreg.v = zero_vreg.v;
  c54_vreg.v = zero_vreg.v;

  __m128d ro2;


#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    _mm_prefetch((const char *)(ob + (p+8)*ldb), _MM_HINT_T0);
    _mm_prefetch((const char *)(ob + (p+8)*ldb + 4), _MM_HINT_T0);
    _mm_prefetch((const char *)a+48, _MM_HINT_T0);

    double  *b_ij_pntr = ob + p * ldb;
    
    b00_vreg.v = _mm256_load_pd(b_ij_pntr);
    b04_vreg.v = _mm256_load_pd(b_ij_pntr + 4);
    
    //b00_vreg.v = (__m256d)_mm256_stream_load_si256((__m256i*)b_ij_pntr);
    //b04_vreg.v = (__m256d)_mm256_stream_load_si256((__m256i*)(b_ij_pntr+4));

//    a0_vreg.v = _mm256_load_pd( (double *) a );       /* load and duplicate */
//    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);
    c04_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c04_vreg.v);
//        if(outi==2016 && outj == 352){
//           print256(" c00_vreg ", c00_vreg);
//        }

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);
    c14_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c14_vreg.v);

    _mm256_store_pd(b_to, b00_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a+2));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);
    c24_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c24_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);
    c34_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c34_vreg.v);

    _mm256_store_pd(b_to+4, b04_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+4) );   /* load and duplicate */
    ro2 = _mm_load_pd((double*)(a+4));
    a0_vreg.v = _mm256_set_m128d(ro2, ro2);
    c40_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c40_vreg.v);
    c44_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c44_vreg.v);

//    a0_vreg.v = _mm256_set1_pd( *(double *) (a+5) );   /* load and duplicate */
     a0_vreg.v = _mm256_permute_pd(a0_vreg.v, 0b0101);
    c50_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c50_vreg.v);
    c54_vreg.v = _mm256_fmadd_pd(b04_vreg.v, a0_vreg.v, c54_vreg.v);

    a += 6;
    b_to += 8;

/*
  if(profiling)
  {
     print256(" in computing a ", a_0p_a_1p_a_2p_a_3p_vreg);
     print256(" in computing b ", b_p0_vreg);
     print256(" in computing c ", c_00_c_10_c_20_c_30_vreg);
  }
*/

  }
/* 
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
  c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 
  c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
  c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(&C(4,4))); 
  c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
  c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(&C(5,4)));


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(0, 4), c04_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(1, 4), c14_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(2, 4), c24_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);
  _mm256_store_pd(&C(3, 4), c34_vreg.v);
  _mm256_store_pd(&C(4, 0), c40_vreg.v);
  _mm256_store_pd(&C(4, 4), c44_vreg.v);
  _mm256_store_pd(&C(5, 0), c50_vreg.v);
  _mm256_store_pd(&C(5, 4), c54_vreg.v);
*/

    if(lastKC)
    {  
      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 
      c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(packedC+32));
      c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(packedC+36)); 
      c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(packedC+40));
      c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(packedC+44));


      b00_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c00_vreg.v, c10_vreg.v, 0b0101);
      c00_vreg.v = b00_vreg.v;
      c10_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c04_vreg.v, c14_vreg.v, 0b0101);
      c04_vreg.v = b00_vreg.v;
      c14_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c20_vreg.v, c30_vreg.v, 0b0101);
      c20_vreg.v = b00_vreg.v;
      c30_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c24_vreg.v, c34_vreg.v, 0b0101);
      c24_vreg.v = b00_vreg.v;
      c34_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c40_vreg.v, c50_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c40_vreg.v, c50_vreg.v, 0b0101);
      c40_vreg.v = b00_vreg.v;
      c50_vreg.v = b04_vreg.v;
      b00_vreg.v = _mm256_blend_pd(c44_vreg.v, c54_vreg.v, 0b1010);
      b04_vreg.v = _mm256_blend_pd(c44_vreg.v, c54_vreg.v, 0b0101);
      c44_vreg.v = b00_vreg.v;
      c54_vreg.v = b04_vreg.v;



      c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
      c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(&C(0,4)));
      c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
      c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(&C(1,4)));   
      c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
      c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(&C(2,4))); 
      c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
      c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(&C(3,4))); 
      c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
      c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(&C(4,4))); 
      c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
      c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(&C(5,4)));

      _mm256_store_pd(&C(0, 0), c00_vreg.v);
      _mm256_store_pd(&C(0, 4), c04_vreg.v);
      _mm256_store_pd(&C(1, 0), c10_vreg.v);
      _mm256_store_pd(&C(1, 4), c14_vreg.v);
      _mm256_store_pd(&C(2, 0), c20_vreg.v);
      _mm256_store_pd(&C(2, 4), c24_vreg.v);
      _mm256_store_pd(&C(3, 0), c30_vreg.v);
      _mm256_store_pd(&C(3, 4), c34_vreg.v);
      _mm256_store_pd(&C(4, 0), c40_vreg.v);
      _mm256_store_pd(&C(4, 4), c44_vreg.v);
      _mm256_store_pd(&C(5, 0), c50_vreg.v);
      _mm256_store_pd(&C(5, 4), c54_vreg.v);
    }
    else
    
    {
        if(!firstKC)
        {
          c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(packedC+0));   
          c04_vreg.v = _mm256_add_pd(c04_vreg.v, _mm256_load_pd(packedC+4));
          c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(packedC+8)); 
          c14_vreg.v = _mm256_add_pd(c14_vreg.v, _mm256_load_pd(packedC+12));   
          c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(packedC+16)); 
          c24_vreg.v = _mm256_add_pd(c24_vreg.v, _mm256_load_pd(packedC+20)); 
          c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(packedC+24));
          c34_vreg.v = _mm256_add_pd(c34_vreg.v, _mm256_load_pd(packedC+28)); 
          c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(packedC+32));
          c44_vreg.v = _mm256_add_pd(c44_vreg.v, _mm256_load_pd(packedC+36)); 
          c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(packedC+40));
          c54_vreg.v = _mm256_add_pd(c54_vreg.v, _mm256_load_pd(packedC+44));
        }

//      print256(" c10_vreg ", c10_vreg);
//      print256(" c04_vreg ", c04_vreg);

//        if(outi==2016 && outj == 352 && !firstKC)
//           printf(" before. packedC[0] = %f \n", packedC[0]);

      _mm256_store_pd(packedC+0, c00_vreg.v);
      _mm256_store_pd(packedC+4, c04_vreg.v);
      _mm256_store_pd(packedC+8, c10_vreg.v);
      _mm256_store_pd(packedC+12, c14_vreg.v);
      _mm256_store_pd(packedC+16, c20_vreg.v);
      _mm256_store_pd(packedC+20, c24_vreg.v);
      _mm256_store_pd(packedC+24, c30_vreg.v);
      _mm256_store_pd(packedC+28, c34_vreg.v);
      _mm256_store_pd(packedC+32, c40_vreg.v);
      _mm256_store_pd(packedC+36, c44_vreg.v);
      _mm256_store_pd(packedC+40, c50_vreg.v);
      _mm256_store_pd(packedC+44, c54_vreg.v);
/*
        if(outi==2016 && outj == 352 && !firstKC){
            printf(" firstKC = %d  \n", firstKC);
           print256(" ----------------------------- c00_vreg  ", c00_vreg);
           printf(" after.  packedC[0] = %f \n", packedC[0]);
        }
*/
    }

}




