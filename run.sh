#!/bin/bash
EXECUTABLE=$1
env LD_PRELOAD=./build/libcuProfile.so LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd` $1
#/usr/local/cuda/extras/CUPTI/samples/profiling_injection/simple_target