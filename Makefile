


 MKL_FLAG= -mkl -D_MKL_ -O3 -march=core-avx2 -DD=2048

 OPT_FLAG= -lopenblas -D_OPENBLAS_ -march=core-avx2 -O3 -DD=2048
 DEBUG_FLAG= -lopenblas -D_OPENBLAS_ -march=core-avx2 -O0 -DD=2048

all: gemm.cpp
	icpc $(OPT_FLAG) gemm.cpp -o gemm
mkl: gemm.cpp
	icpc $(MKL_FLAG) gemm.cpp -o gemm
thread: gemm.cpp
	icpc $(OPT_FLAG) gemm.cpp -o gemm -fopenmp
debug:
	icpc $(DEBUG_FLAG) gemm.cpp -g -o gemm
assemble:
	icpc $(OPT_FLAG) gemm.cpp -o gemm.s -S
clean:
	rm gemm
