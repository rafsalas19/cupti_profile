#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../include/cuptiErrorCheck.h"
#include "../include/profileSession.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvperf_host.h"
#include <vector>
#include <mutex>
#include <string>
#include "cupti_driver_cbid.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"




using namespace std;

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

    // Initialize fields, with env var overrides
    ctxProfilerData() : curRanges(), maxRangeNameLength(64), iterations()
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

ProfileSession::ProfileSession(){}
	
void ProfileSession::callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata){
	
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API){
		cout << "Hello world from the injection library CB: CUPTI_CB_DOMAIN_DRIVER_API" << endl;
	}
	else if (domain == CUPTI_CB_DOMAIN_RESOURCE){
	
		if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)// contex should be created by default on the first CUDA runtime API call
        {
			CUpti_ResourceData const * res_data = static_cast<CUpti_ResourceData const *>(cbdata);
            CUcontext ctx = res_data->context;
			ctxProfilerData data = { };
			data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(data.dev_id)));

            RUNTIME_API_CALL(cudaGetDeviceProperties(&(data.dev_prop), data.dev_id));

			ctx_data_mutex.lock();
			if (data.dev_prop.major >= 7) //check compute capability 
            {
                ctx_data[ctx] = data;
                initialize_ctx_data(ctx_data[ctx]);
            }
            else if (ctx_data.count(ctx))
            {
                ctx_data.erase(ctx);
            }
			ctx_data_mutex.unlock();
		}
		
	  
		cout << "Hello world from the injection library CB: CUPTI_CB_DOMAIN_RESOURCE" << endl;
	}		
}

void ProfileSession::startSession(){}

void ProfileSession::endSession(){}

void ProfileSession::exitCB(){}

void ProfileSession::subscribeCB(){
		CUpti_SubscriberHandle subscriber;
    	CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)(callback), NULL));
    	CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    	CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    	
	// Register callback for application exit
    	atexit(exitCB);	
}


