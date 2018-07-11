


# OPT_FLAG= -mkl -qopenmp -D_MKL_ -O3 -march=core-avx2

 OPT_FLAG= -lopenblas -D_OPENBLAS_ -march=core-avx2 -O3 -DD=2048

all: gemm.cpp
	icpc $(OPT_FLAG) gemm.cpp -o gemm
debug:
	icpc -O3 cnn_test.cpp mat.cpp -o cnn.test.g -g -lopenblas
clean:
	rm cnn.test *.o conv
