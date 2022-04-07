#Linux only build

CUDA_DIR=/usr/local/cuda
INCLUDES= -I ./include -I $(CUDA_DIR)/extras/CUPTI/include
SRC=./src
OUTDIR= ./build
LIB_PATH=$(CUDA_DIR)/lib64 
CUPTI_LIB_PATH=$(CUDA_DIR)/extras/CUPTI/lib64
LIBS= -L $(LIB_PATH) -L $(CUPTI_LIB_PATH)
NVCCFLAGS := -lcuda  -lcupti -lnvperf_host -lnvperf_target -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 --std c++11 
BUILD_DIR=@mkdir -p $(OUTDIR)i


MKDIR_P = mkdir -p

.PHONY: directories

all: directories cupti_profile libcuProfile.so

directories: ${OUTDIR}

${OUTDIR}:
	${MKDIR_P} ${OUTDIR}

cupti_profile: main.o 
	nvcc $(NVCCFLAGS) -o $(OUTDIR)/$@ $(OUTDIR)/$^ $(LIBS) $(INCLUDES)

main.o: main.cu
	nvcc $(NVCCFLAGS) -c $(INCLUDES) $(LIBS) $< -o $(OUTDIR)/$@

libcuProfile.so:$(SRC)/* ./include/profileSession.h ./include/cuptiMetrics.h ./include/utils.h
	nvcc $(SRC)/* -lcuda -lcupti -lnvperf_host -lnvperf_target  $(INCLUDES) $(LIBS) -Ldl -Xcompiler -fPIC --shared -o $(OUTDIR)/$@ 

clean:
	rm -f $(OUTDIR)/cupti_profile $(OUTDIR)/*.o  $(OUTDIR)/*.so
