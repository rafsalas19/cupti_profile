#!/bin/bash
#export set  CUPTI_METRIC_CODES=1000,1022
EXECUTABLE=$1
LD_LIBRARY_PATH=:/usr/local/cuda/extras/CUPTI/lib64/$LD_LIBRARY_PATH
env LD_PRELOAD=./build/libcuProfile.so LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd` $EXECUTABLE
#/usr/local/cuda/extras/CUPTI/samples/profiling_injection/simple_target
