


/* Create macros so that the matrices are stored in column-major order */
/*
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
*/

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]
#define packedC(i,j) packedC[ (i)*ldc + (j) ]



#define Bs(i,j) Bs[ (j)*ldb + (i) ]


typedef union
{
  __m256d v;
  double d[4];
} v4df_t;



void AddDot4x4( int, double *, int, double *, int, double *, int);
void AddDot4x8( int, double *, int, double *, int, double *, int);
void AddDot12x4( int, double *, int, double *, int, double *, int);

void PackB_and_AddDot12x4( int, double *, int, double *, int, double *, int, double *, int);

void PackMatrixA_4( int, double *, int, double *, int, int);
void PackMatrixA_12( int, double *, int, double *, int, int);

void PackMatrixB_12( int, double *, int, double * );
void PackMatrixB_4( int, double *, int, double * );

void InnerKernel12x4( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time, double* packedA, double* packedB)
{
  int i, j;

  for ( j=0; j<m; j+=12 ){        /* Loop over the columns of C, unrolled by 4 */
    if ( first_time )
    {
        if(j!=60)
            PackMatrixA_12( k, &A( j, 0 ), lda, &packedA[ j*k ], j, m);
        else
            PackMatrixA_4( k, &A( j, 0 ), lda, &packedA[ j*k ], j, m);

    }
    for ( i=0; i<n; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if ( j == 0 ) 
      {
//	      PackMatrixB( k, &B( 0, i ), ldb, &packedB[ i*k ]);
          PackB_and_AddDot12x4( k, &packedA[ j*k ], 6, &B( 0, i ), ldb, &packedB[ i*k ], 8, &C( j,i ), ldc);
//          AddDot6x8( k, &packedA[ j*k ], 6, &packedB[ i*k ], 8, &C( j,i ), ldc);
      }
      else if(j==60)
          AddDot4x4( k, &packedA[ j*k ], 6, &packedB[ i*k ], 8, &C( j,i ), ldc);
      else
          AddDot12x4( k, &packedA[ j*k ], 6, &packedB[ i*k ], 8, &C( j,i ), ldc);
    }
  }
}

void PackMatrixB_12( int k, double *b, int ldb, double *b_to)
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
    
    reg1 = _mm256_load_pd(b_ij_pntr + 8);
    _mm256_store_pd(b_to, reg1);
    b_to += 4;

  }
}


void PackMatrixA_12( int k, double *a, int lda, double *a_to, int j, int m)
{
  int i;
      double 
        *a_i0_pntr = &A( 0, 0 ), *a_i1_pntr = &A( 1, 0 ),
        *a_i2_pntr = &A( 2, 0 ), *a_i3_pntr = &A( 3, 0 ),
        *a_i4_pntr = &A( 4, 0 ), *a_i5_pntr = &A( 5, 0 ),
        *a_i6_pntr = &A( 6, 0 ), *a_i7_pntr = &A( 6, 0 ),
        *a_i8_pntr = &A( 8, 0 ), *a_i9_pntr = &A( 9, 0 ),
        *a_i10_pntr = &A( 10, 0 ), *a_i11_pntr = &A( 11, 0 );

      v4df_t vreg;
#pragma unroll(4)
      for( i=0; i<k; i++){  /* loop over rows of B */
        *a_to++ = *a_i0_pntr++;
        *a_to++ = *a_i1_pntr++;
        *a_to++ = *a_i2_pntr++;
        *a_to++ = *a_i3_pntr++;
        *a_to++ = *a_i4_pntr++;
        *a_to++ = *a_i5_pntr++;
        *a_to++ = *a_i6_pntr++;
        *a_to++ = *a_i7_pntr++;
        *a_to++ = *a_i8_pntr++;
        *a_to++ = *a_i9_pntr++;
        *a_to++ = *a_i10_pntr++;
        *a_to++ = *a_i11_pntr++;
      }
}


void PackMatrixB_4( int k, double *b, int ldb, double *b_to)
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

  }
}


void PackMatrixA_4( int k, double *a, int lda, double *a_to, int j, int m)
{
  int i;
      double 
        *a_i0_pntr = &A( 0, 0 ), *a_i1_pntr = &A( 1, 0 ),
        *a_i2_pntr = &A( 2, 0 ), *a_i3_pntr = &A( 3, 0 );

      v4df_t vreg;
#pragma unroll(4)
      for( i=0; i<k; i++){  /* loop over rows of B */
        *a_to++ = *a_i0_pntr++;
        *a_to++ = *a_i1_pntr++;
        *a_to++ = *a_i2_pntr++;
        *a_to++ = *a_i3_pntr++;
      }
}



void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc)
{

  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c04_vreg,
      c10_vreg, c14_vreg,
      c20_vreg, c24_vreg,
      c30_vreg, c34_vreg,

    b00_vreg,

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


#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){

    b00_vreg.v = _mm256_load_pd( (double *) b );
    b += 4;


    /* First row and second rows */

    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);

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
 
//    _mm_prefetch((const char *)b+k*8, _MM_HINT_T1);
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);


}


void PackB_and_AddDot12x4( int k, double *a, int lda, double *ob, int ldb,  double *b, int ldb2, double *c, int ldc)
{
  int j;

  double* b_to = b;
  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c60_vreg,
      c10_vreg, c70_vreg,
      c20_vreg, c80_vreg,
      c30_vreg, c90_vreg,
      c40_vreg, c100_vreg,
      c50_vreg, c110_vreg,

    b00_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);

  c00_vreg.v = zero_vreg.v; 
  c60_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c70_vreg.v = zero_vreg.v;
  c20_vreg.v = zero_vreg.v;
  c80_vreg.v = zero_vreg.v;
  c30_vreg.v = zero_vreg.v;
  c90_vreg.v = zero_vreg.v;
  c40_vreg.v = zero_vreg.v;
  c100_vreg.v = zero_vreg.v;
  c50_vreg.v = zero_vreg.v;
  c110_vreg.v = zero_vreg.v;


#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    _mm_prefetch((const char *)(ob + (p+1)*ldb), _MM_HINT_T1);
    _mm_prefetch((const char *)a+8, _MM_HINT_T1);

    double  *b_ij_pntr = ob + p * ldb;
    
    b00_vreg.v = _mm256_load_pd(b_ij_pntr);

//    b00_vreg.v = _mm256_load_pd( (double *) b );
//    b04_vreg.v = _mm256_load_pd( (double *) (b+4) );
//    b += 8;


    /* First row and second rows */

    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);

    _mm256_store_pd(b_to, b00_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+4) );   /* load and duplicate */
    c40_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c40_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+5) );   /* load and duplicate */
    c50_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c50_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+6) );   /* load and duplicate */
    c60_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c60_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+7) );   /* load and duplicate */
    c70_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c70_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+8) );   /* load and duplicate */
    c80_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c80_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+9) );   /* load and duplicate */
    c90_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c90_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+10) );   /* load and duplicate */
    c100_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c100_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+11) );   /* load and duplicate */
    c110_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c110_vreg.v);




    a += 12;
    b_to += 4;

/*
  if(profiling)
  {
     print256(" in computing a ", a_0p_a_1p_a_2p_a_3p_vreg);
     print256(" in computing b ", b_p0_vreg);
     print256(" in computing c ", c_00_c_10_c_20_c_30_vreg);
  }
*/

  }
 
//    _mm_prefetch((const char *)b+k*8, _MM_HINT_T1);
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c60_vreg.v = _mm256_add_pd(c60_vreg.v, _mm256_load_pd(&C(6,0)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c70_vreg.v = _mm256_add_pd(c70_vreg.v, _mm256_load_pd(&C(7,0)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c80_vreg.v = _mm256_add_pd(c80_vreg.v, _mm256_load_pd(&C(8,0))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
  c90_vreg.v = _mm256_add_pd(c90_vreg.v, _mm256_load_pd(&C(9,0))); 
  c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
  c100_vreg.v = _mm256_add_pd(c100_vreg.v, _mm256_load_pd(&C(10,0))); 
  c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
  c110_vreg.v = _mm256_add_pd(c110_vreg.v, _mm256_load_pd(&C(11,0)));


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(6, 0), c60_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(7, 0), c70_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(8, 0), c80_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);
  _mm256_store_pd(&C(9, 0), c90_vreg.v);
  _mm256_store_pd(&C(4, 0), c40_vreg.v);
  _mm256_store_pd(&C(10, 0), c100_vreg.v);
  _mm256_store_pd(&C(5, 0), c50_vreg.v);
  _mm256_store_pd(&C(11, 0), c110_vreg.v);


}



void AddDot12x4( int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int j;

  int p;
  v4df_t
      zero_vreg,

      c00_vreg, c60_vreg,
      c10_vreg, c70_vreg,
      c20_vreg, c80_vreg,
      c30_vreg, c90_vreg,
      c40_vreg, c100_vreg,
      c50_vreg, c110_vreg,

    b00_vreg,

    a0_vreg;

  zero_vreg.v = _mm256_set1_pd(0);

  c00_vreg.v = zero_vreg.v; 
  c60_vreg.v = zero_vreg.v;
  c10_vreg.v = zero_vreg.v;
  c70_vreg.v = zero_vreg.v;
  c20_vreg.v = zero_vreg.v;
  c80_vreg.v = zero_vreg.v;
  c30_vreg.v = zero_vreg.v;
  c90_vreg.v = zero_vreg.v;
  c40_vreg.v = zero_vreg.v;
  c100_vreg.v = zero_vreg.v;
  c50_vreg.v = zero_vreg.v;
  c110_vreg.v = zero_vreg.v;


#pragma noprefetch 
#pragma unroll(8)
  for ( p=0; p<k; p++ ){
    
    b00_vreg.v = _mm256_load_pd( (double *) b );
    b += 4;


    /* First row and second rows */

    a0_vreg.v = _mm256_set1_pd( *(double *) a );       /* load and duplicate */
    c00_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c00_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+1) );   /* load and duplicate */
    c10_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c10_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+2) );   /* load and duplicate */
    c20_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c20_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+3) );   /* load and duplicate */
    c30_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c30_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+4) );   /* load and duplicate */
    c40_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c40_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+5) );   /* load and duplicate */
    c50_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c50_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+6) );   /* load and duplicate */
    c60_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c60_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+7) );   /* load and duplicate */
    c70_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c70_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+8) );   /* load and duplicate */
    c80_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c80_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+9) );   /* load and duplicate */
    c90_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c90_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+10) );   /* load and duplicate */
    c100_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c100_vreg.v);

    a0_vreg.v = _mm256_set1_pd( *(double *) (a+11) );   /* load and duplicate */
    c110_vreg.v = _mm256_fmadd_pd(b00_vreg.v, a0_vreg.v, c110_vreg.v);

    a += 12;

/*
  if(profiling)
  {
     print256(" in computing a ", a_0p_a_1p_a_2p_a_3p_vreg);
     print256(" in computing b ", b_p0_vreg);
     print256(" in computing c ", c_00_c_10_c_20_c_30_vreg);
  }
*/

  }
 
//    _mm_prefetch((const char *)b+k*8, _MM_HINT_T1);
  c00_vreg.v = _mm256_add_pd(c00_vreg.v, _mm256_load_pd(&C(0,0)));   
  c60_vreg.v = _mm256_add_pd(c60_vreg.v, _mm256_load_pd(&C(6,0)));
  c10_vreg.v = _mm256_add_pd(c10_vreg.v, _mm256_load_pd(&C(1,0))); 
  c70_vreg.v = _mm256_add_pd(c70_vreg.v, _mm256_load_pd(&C(7,0)));   
  c20_vreg.v = _mm256_add_pd(c20_vreg.v, _mm256_load_pd(&C(2,0))); 
  c80_vreg.v = _mm256_add_pd(c80_vreg.v, _mm256_load_pd(&C(8,0))); 
  c30_vreg.v = _mm256_add_pd(c30_vreg.v, _mm256_load_pd(&C(3,0)));
  c90_vreg.v = _mm256_add_pd(c90_vreg.v, _mm256_load_pd(&C(9,0))); 
  c40_vreg.v = _mm256_add_pd(c40_vreg.v, _mm256_load_pd(&C(4,0)));
  c100_vreg.v = _mm256_add_pd(c100_vreg.v, _mm256_load_pd(&C(10,0))); 
  c50_vreg.v = _mm256_add_pd(c50_vreg.v, _mm256_load_pd(&C(5,0)));
  c110_vreg.v = _mm256_add_pd(c110_vreg.v, _mm256_load_pd(&C(11,0)));


  _mm256_store_pd(&C(0, 0), c00_vreg.v);
  _mm256_store_pd(&C(6, 0), c60_vreg.v);
  _mm256_store_pd(&C(1, 0), c10_vreg.v);
  _mm256_store_pd(&C(7, 0), c70_vreg.v);
  _mm256_store_pd(&C(2, 0), c20_vreg.v);
  _mm256_store_pd(&C(8, 0), c80_vreg.v);
  _mm256_store_pd(&C(3, 0), c30_vreg.v);
  _mm256_store_pd(&C(9, 0), c90_vreg.v);
  _mm256_store_pd(&C(4, 0), c40_vreg.v);
  _mm256_store_pd(&C(10, 0), c100_vreg.v);
  _mm256_store_pd(&C(5, 0), c50_vreg.v);
  _mm256_store_pd(&C(11, 0), c110_vreg.v);


}




