#Linux only build

CUDA_DIR=/usr/local/cuda
INCLUDES= -I ./include -I $(CUDA_DIR)/extras/CUPTI/include
SRC=./src
OUTDIR= ./build
LIB_PATH=$(CUDA_DIR)/lib64 
CUPTI_LIB_PATH=$(CUDA_DIR)/extras/CUPTI/lib64
LIBS= -L $(LIB_PATH) -L $(CUPTI_LIB_PATH)
NVCCFLAGS := -lcuda  -lcupti -lnvperf_host -lnvperf_target - --std c++11 
#gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
BUILD_DIR=@mkdir -p $(OUTDIR)

$(info    SRC is $(SRC))
$(info    OUTDIR is $(OUTDIR))
$(info    INCLUDES is $(INCLUDES))
$(info    LIB_PATH is $(LIB_PATH))

MKDIR_P = mkdir -p

.PHONY: directories

all: directories cupti_profile libcuProfile.so

directories: ${OUTDIR}

${OUTDIR}:
	${MKDIR_P} ${OUTDIR}

cupti_profile: main.o utils.o cuptiMetrics.o
	nvcc $(nvccflags) -o $(OUTDIR)/$@ $(OUTDIR)/main.o $(OUTDIR)/cuptiMetrics.o $(OUTDIR)/utils.o $(LIBS) $(INCLUDES) -lnvperf_host -lnvperf_target -Xcompiler -fpic

main.o: main.cu
	nvcc $(nvccflags) -c $(INCLUDES) $(LIBS) $< -o $(OUTDIR)/$@  -Xcompiler -fpic

libcuProfile.so: $(SRC)/PWMetrics.cpp  $(SRC)/cuptiMetrics.cpp  $(SRC)/injection.cpp  $(SRC)/profileSession.cpp utils.o cuptiMetrics.o
	nvcc $(SRC)/PWMetrics.cpp $(SRC)/injection.cpp  $(SRC)/profileSession.cpp -lcuda -lcupti -lnvperf_host -lnvperf_target  $(INCLUDES) $(LIBS) $(OUTDIR)/cuptiMetrics.o $(OUTDIR)/utils.o -ldl -Xcompiler -fpic --shared -o $(OUTDIR)/$@ 
	
utils.o: $(SRC)/utils.cpp  
	nvcc -c $(SRC)/utils.cpp $(nvccflags) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 

cuptiMetrics.o: $(SRC)/cuptiMetrics.cpp  
	nvcc -c $(SRC)/cuptiMetrics.cpp   $(nvccflags) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 


clean:
	rm -f $(OUTDIR)/cupti_profile $(OUTDIR)/*.o  $(OUTDIR)/*.so

# cupti_profile: main.o 
	# nvcc $(NVCCFLAGS) -o $(OUTDIR)/$@ $(OUTDIR)/$^ $(LIBS) $(INCLUDES)

# main.o: main.cu
	# nvcc $(NVCCFLAGS) -c $(INCLUDES) $(LIBS) $< -o $(OUTDIR)/$@

# libcuProfile.so:$(SRC)/* ./include/profileSession.h ./include/cuptiMetrics.h ./include/utils.h ./include/PWMetrics.h
	# nvcc $(SRC)/* -lcuda -lcupti -lnvperf_host -lnvperf_target  $(INCLUDES) $(LIBS) -Ldl -Xcompiler -fPIC --shared -o $(OUTDIR)/$@ 

# clean:
	# rm -f $(OUTDIR)/cupti_profile $(OUTDIR)/*.o  $(OUTDIR)/*.so