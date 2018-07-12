


# OPT_FLAG= -mkl -qopenmp -D_MKL_ -O3 -march=core-avx2

 OPT_FLAG= -lopenblas -D_OPENBLAS_ -march=core-avx2 -O3 -DD=2048

all: gemm.cpp
	icpc $(OPT_FLAG) gemm.cpp -o gemm
debug:
	icpc $(DEBUG_FLAG) gemm.cpp -o -g gemm
assemble:
	icpc $(OPT_FLAG) gemm.cpp -o gemm.s -S
clean:
	rm gemm
