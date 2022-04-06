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
#include "../include/cuptiMetrics.h"



using namespace std;





void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata){
	
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API){
		cout << "Hello world from the injection library CB: CUPTI_CB_DOMAIN_DRIVER_API" << endl;
	}
	else if (domain == CUPTI_CB_DOMAIN_RESOURCE){
	
		if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)// contex should be created by default on the first CUDA runtime API call
		{
			for(int i =0;i<metricNames.size();++i){
				std::cout<<metricNames[i]<<endl;
			}
			CUpti_ResourceData const * res_data = static_cast<CUpti_ResourceData const *>(cbdata);
            CUcontext ctx = res_data->context;
			ctxProfilerData data = { };
			data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(data.dev_id)));

            RUNTIME_API_CALL(cudaGetDeviceProperties(&(data.dev_prop), data.dev_id));

			ctx_data_mutex.lock();
			if (data.dev_prop.major >= 7) //check compute capability 
            {
               ctx_data_map[ctx] = data;
               //initialize_ctx_data(ctx_data_map[ctx]);
			   //initialize_ctx_data();
            }
            else if (ctx_data_map.count(ctx))
            {
                ctx_data_map.erase(ctx);
            }
			ctx_data_mutex.unlock();
		}
		
	  
		cout << "Hello world from the injection library CB: CUPTI_CB_DOMAIN_RESOURCE" << endl;
	}		
}

void startSession(){}

void endSession(){}

void exitCB(){}

void subscribeCB(){
		CUpti_SubscriberHandle subscriber;
    	CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)(callback), NULL));
    	CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    	CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    	
	// Register callback for application exit
    	atexit(exitCB);	
}