#ifndef PROF_SESSION_H
#define PROF_SESSION_H

#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"
#include <stdlib.h>

#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include<utility>
#include "../include/cuptiMetrics.h"
using namespace std;

struct MetricRecord;

//holds all context data
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
	vector <MetricRecord> results;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
	int				maxSessions;
    int             iterations; // Count of sessions
	CUpti_ProfilerRange profilerRange;// CUPTI_AutoRange or CUPTI_UserRange;

    // Initialize fields, with env var overrides
    ctxProfilerData() : curRanges(), maxRangeNameLength(128), iterations(),profilerRange(CUPTI_AutoRange)
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
            maxNumRanges = 3;
        }
		maxSessions =1;
    };
};

//metric handler
extern CuptiMetrics cupMetrics;

void startSession(ctxProfilerData &ctx_data);

void endSession(ctxProfilerData &ctx_data);
	
void subscribeCB();
	
void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata);

//what to do at the end of the run
void exitCB();
		
void printContextMetrics(const ctxProfilerData &ctx_data);
	
extern mutex ctx_data_mutex;//protext context map
	
extern unordered_map<CUcontext, ctxProfilerData> ctx_data_map; //holds all contexts

extern CUpti_SubscriberHandle cupti_subscriber;


#endif