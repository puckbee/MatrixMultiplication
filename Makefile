
MKL_FLAG= -mkl -D_MKL_ -O3 -march=core-avx2 -DD=4096

OPT_FLAG= -D_OPENBLAS_ -march=core-avx2 -O3 -DD=4096 -xCORE-AVX2
DEBUG_FLAG= -lopenblas -D_OPENBLAS_ -march=core-avx2 -O0 -DD=2048

SERIAL_FLAG = -lopenblas
PTHREAD_FLAG = -lopenblasp -lpthread -lgfortran -D_SUBDIR_
OPENMP_FLAG = -lopenblaso 

all: gemm.cpp
	icpc $(OPT_FLAG) $(SERIAL_FLAG) gemm.cpp -o gemm

mkl: gemm.cpp
	icpc $(MKL_FLAG) gemm.cpp -o gemm -fopenmp

mt-pthread: gemm.cpp
	icpc $(OPT_FLAG) $(PTHREAD_FLAG) gemm.cpp -o gemm -fopenmp

mt-openmp: gemm.cpp
	icpc $(OPT_FLAG) $(OPENMP_FLAG) gemm.cpp -o gemm -fopenmp

mt-self: gemm.cpp
	icpc $(OPT_FLAG) $(SERIAL_FLAG) gemm.cpp -o gemm -fopenmp

thread: gemm.cpp
	icpc $(OPT_FLAG) gemm.cpp -o gemm -fopenmp -lopenblas

debug:
	icpc $(DEBUG_FLAG) $(SERIAL_FLAG) gemm.cpp -g -o gemm

assemble:
	icpc $(OPT_FLAG) $(SERIAL_FLAG) gemm.cpp -o gemm.s -S

clean:
	rm gemm

