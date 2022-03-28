#Linux only build

CUDA_DIR=/usr/local/cuda
INCLUDES= -I ./include -I $(CUDA_DIR)/extras/CUPTI/include
SRC=./src/
OUTDIR= ./build
LIB_PATH=$(CUDA_DIR)/lib64 
CUPTI_LIB_PATH=$(CUDA_DIR)/extras/CUPTI/lib64
LIBS= -L $(LIB_PATH) -L $(CUPTI_LIB_PATH)
NVCCFLAGS := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 


all: cupti_profile libcuProfile.so

cupti_profile: main.o
	nvcc $(NVCC_COMPILER) $(NVCCFLAGS) -o $(OUTDIR)/$@ $(OUTDIR)/$^ $(LIBS) $(INCLUDES)

main.o: main.cu
	nvcc $(NVCC_COMPILER)  $(NVCCFLAGS) -c $(INCLUDES) $< -o $(OUTDIR)/$@

libcuProfile.so: $(SRC)/injection.cpp
	nvcc -o $(OUTDIR)/$@ $< $(INCLUDES) $(LIBS) -Ldl -Xcompiler -fPIC --shared

clean:
	rm -f $(OUTDIR)/cupti_profile $(OUTDIR)/*.o  $(OUTDIR)/*.so
