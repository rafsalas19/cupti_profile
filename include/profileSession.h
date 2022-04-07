#ifndef PROF_SESSION_H
#define PROF_SESSION_H

#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"
#include <stdlib.h>

#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <vector>
using ::std::vector;

#include <unordered_map>
using ::std::unordered_map;

#include <unordered_set>
using ::std::unordered_set;

#include <iostream>
#include<utility>

using ::std::pair;

struct ctxProfilerData
{
    CUcontext       ctx;
    int             dev_id;
    cudaDeviceProp  dev_prop;
    vector<uint8_t> counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    vector<uint8_t> counterDataImage;
    vector<uint8_t> counterDataPrefixImage;
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
    int             iterations; // Count of sessions
	CUpti_ProfilerRange profilerRange;// CUPTI_AutoRange or CUPTI_UserRange;

    // Initialize fields, with env var overrides
    ctxProfilerData() : curRanges(), maxRangeNameLength(64), iterations(),profilerRange(CUPTI_AutoRange)
    {
        char * env_var = getenv("INJECTION_KERNEL_COUNT");
        if (env_var != NULL)
        {
            int val = atoi(env_var);
            if (val < 1)
            {
                cerr << "Read " << val << " kernels from INJECTION_KERNEL_COUNT, but must be >= 1; defaulting to 10." << endl;
                val = 10;
            }
            maxNumRanges = val;
        }
        else
        {
            maxNumRanges = 10;
        }
    };
};

extern std::vector<std::string> metricNames;

void startSession(ctxProfilerData &ctx_data);

void endSession(ctxProfilerData &ctx_data);
	
void subscribeCB();
	
void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata);

void exitCB();
		
	
void print_context(const ctxProfilerData &ctx_data);
	
extern mutex ctx_data_mutex;
	
extern unordered_map<CUcontext, ctxProfilerData> ctx_data_map;



#endif