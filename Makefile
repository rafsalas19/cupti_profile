#Linux only build

CUDA_DIR=/usr/local/cuda
INCLUDES= -I ./include -I $(CUDA_DIR)/extras/CUPTI/include -I/usr/include/python3.6m
SRC=./src
OUTDIR= ./build
LIB_PATH=$(CUDA_DIR)/lib64 
CUPTI_LIB_PATH=$(CUDA_DIR)/extras/CUPTI/lib64
LIBS= -L $(LIB_PATH) -L $(CUPTI_LIB_PATH)
NVCCFLAGS := -lcuda  -lcupti -lnvperf_host -lnvperf_target --std c++11 
#gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
BUILD_DIR=@mkdir -p $(OUTDIR)

$(info    SRC is $(SRC))
$(info    OUTDIR is $(OUTDIR))
$(info    INCLUDES is $(INCLUDES))
$(info    LIB_PATH is $(LIB_PATH))

MKDIR_P = mkdir -p

.PHONY: directories

all: directories cupti_profile libcuProfile.so cupProf.o

directories: ${OUTDIR}

${OUTDIR}:
	${MKDIR_P} ${OUTDIR}

cupti_profile: main.o utils.o cuptiMetrics.o cupProf.o profileSession.o
	nvcc $(NVCCFLAGS) -o $(OUTDIR)/$@ $(OUTDIR)/main.o $(OUTDIR)/utils.o $(OUTDIR)/cuptiMetrics.o $(OUTDIR)/cupProf.o $(OUTDIR)/profileSession.o $(LIBS) $(INCLUDES) -lcuda -lcupti -lnvperf_host -lnvperf_target -Xcompiler -fpic

main.o: main.cu 
	nvcc $(NVCCFLAGS) -c $(INCLUDES) $(LIBS) $< -o $(OUTDIR)/$@ -lcuda -lcupti -lnvperf_host -lnvperf_target -Xcompiler -fpic

libcuProfile.so: $(SRC)/cuptiMetrics.cpp  $(SRC)/injection.cpp profileSession.o utils.o cuptiMetrics.o PWMetrics.o
	nvcc $(SRC)/injection.cpp  -lcuda -lcupti -lnvperf_host -lnvperf_target  $(INCLUDES) $(LIBS) $(OUTDIR)/PWMetrics.o $(OUTDIR)/profileSession.o $(OUTDIR)/cuptiMetrics.o $(OUTDIR)/utils.o -ldl -Xcompiler -fpic --shared -o $(OUTDIR)/$@ 

profileSession.o: $(SRC)/profileSession.cpp
	nvcc -c $(SRC)/profileSession.cpp $(NVCCFLAGS) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 

PWMetrics.o: $(SRC)/PWMetrics.cpp  
	nvcc -c $(SRC)/PWMetrics.cpp $(NVCCFLAGS) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 
	
utils.o: $(SRC)/utils.cpp  
	nvcc -c $(SRC)/utils.cpp $(NVCCFLAGS) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 

cuptiMetrics.o: $(SRC)/cuptiMetrics.cpp  
	nvcc -c $(SRC)/cuptiMetrics.cpp   $(NVCCFLAGS) $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@ 

cupProf.o: $(SRC)/cupProf/cupProf.cpp cuptiMetrics.o profileSession.o utils.o PWMetrics.o
	swig -c++ -python -outdir build/ ./src/swigTest.i
	nvcc -c $(SRC)/cupProf/cupProf.cpp $(NVCCFLAGS)  $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/$@
	nvcc -c $(SRC)/swigTest_wrap.cxx  $(INCLUDES) $(LIBS) -Xcompiler -fpic -o $(OUTDIR)/swigTest_wrap.o
	nvcc $(OUTDIR)/swigTest_wrap.o $(OUTDIR)/cupProf.o $(NVCCFLAGS) -Xcompiler -fpic --shared -o $(OUTDIR)/_cupProf.so  $(INCLUDES) $(LIBS) $(OUTDIR)/cuptiMetrics.o $(OUTDIR)/profileSession.o $(OUTDIR)/utils.o $(OUTDIR)/PWMetrics.o

clean:
	rm -f $(OUTDIR)/cupti_profile $(OUTDIR)/*.o  $(OUTDIR)/*.so $(OUTDIR)/*.a $(OUTDIR)/*.py