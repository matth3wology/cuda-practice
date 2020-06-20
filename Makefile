CC=nvcc
OUTFILE=main
INCLUDE=-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include"
LIB=-L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib"
LIBFLAGS=-lcublas -lcuda

all:
	$(CC) src/main.cu $(INCLUDE) $(LIB) $(LIBFLAGS) $(INCLUDEFLAGS) -o $(OUTFILE)

clean:
	DEL *.ex*
	DEL *.lib
