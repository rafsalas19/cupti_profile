#include <stdio.h>
#include <stdlib.h>
#include <iostream>
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
#include "../include/PWMetrics.h"
#include "../include/utils.h"
#include "../include/cuptiErrorCheck.h"
#include "../include/profileSession.h"

using namespace std;

mutex ctx_data_mutex;

unordered_map<CUcontext, ctxProfilerData> ctx_data_map;

CuptiMetrics cupMetrics;

CUpti_SubscriberHandle cupti_subscriber;

namespace ProfileSession{

	void printContextMetrics(const ctxProfilerData &ctx_data){
		cout << endl << "Context " << ctx_data.ctx << ", device " << ctx_data.dev_id << " (" << ctx_data.dev_prop.name << ") session " << ctx_data.iterations << ":" << endl;
		cupMetrics.printMetricRecords(ctx_data);
	}

	void initialize_ctx_data(ctxProfilerData &ctx_data){
		// CUPTI Profiler API + NVPWinitialization
		try{
			
			static int profiler_initialized = 0;
			if (profiler_initialized == 0){//init only once
				CUpti_Profiler_Initialize_Params profInitParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
				CUPTI_API_CALL(cuptiProfilerInitialize(&profInitParams));
				NVPW_InitializeHost_Params initHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
				NVPW_API_CALL(NVPW_InitializeHost(&initHostParams));
				profiler_initialized = 1;
			}
			
			// Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data
			CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
			getCounterAvailabilityParams.ctx = ctx_data.ctx;
			CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
			// Allocate sized counterAvailabilityImage
			ctx_data.counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
			 // Initialize counterAvailabilityImage
			getCounterAvailabilityParams.pCounterAvailabilityImage = ctx_data.counterAvailabilityImage.data();
			CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
			
			if(!cupMetrics.configureConfigImage(ctx_data)){
				throw std::runtime_error("Failed to create configImage/counterDataPrefixImage for context");
			}
			
			// Record counterDataPrefixImage info and other options for sizing the counterDataImage
			ctx_data.counterDataImageOptions.pCounterDataPrefix = ctx_data.counterDataPrefixImage.data();
			ctx_data.counterDataImageOptions.counterDataPrefixSize = ctx_data.counterDataPrefixImage.size();
			ctx_data.counterDataImageOptions.maxNumRanges = ctx_data.maxNumRanges;
			ctx_data.counterDataImageOptions.maxNumRangeTreeNodes = ctx_data.maxNumRanges;
			ctx_data.counterDataImageOptions.maxRangeNameLength = ctx_data.maxRangeNameLength;

			// Calculate size of counterDataImage based on counterDataPrefixImage and options
			CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
			calculateSizeParams.pOptions = &(ctx_data.counterDataImageOptions);
			calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
			CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
			// Create counterDataImage
			ctx_data.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

			// Initialize counterDataImage inside start_session
			CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
			initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
			initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
			initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
			initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
			CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

			// Calculate scratchBuffer size based on counterDataImage size and counterDataImage
			CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
			scratchBufferSizeParams.counterDataImageSize = ctx_data.counterDataImage.size();
			scratchBufferSizeParams.pCounterDataImage = ctx_data.counterDataImage.data();
			CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
			// Create counterDataScratchBuffer
			ctx_data.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

			// Initialize counterDataScratchBuffer
			CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
			initScratchBufferParams.counterDataImageSize = ctx_data.counterDataImage.size();
			initScratchBufferParams.pCounterDataImage = ctx_data.counterDataImage.data();
			initScratchBufferParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();;
			initScratchBufferParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
			CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));
			
		
		}
		catch(exception& e ){
			cout << e.what() << endl;
			exit(EXIT_FAILURE);
		}
		catch(... ){
			cout << "unknown failure" << endl;
			exit(EXIT_FAILURE);
		}
	}

	void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata){
		
		if (domain == CUPTI_CB_DOMAIN_DRIVER_API){
			//kernel launch
			if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
			{
				CUpti_CallbackData const * data = static_cast<CUpti_CallbackData const *>(cbdata);
				CUcontext ctx = data->context;
				// On entry, enable / update profiling as needed
				if (data->callbackSite == CUPTI_API_ENTER)
				{
					ctx_data_mutex.lock();
					if (ctx_data_map.count(ctx) > 0)//look for the first context data structure as it should have been created if compute capability >=7.0
					{
						// If at maximum number of ranges, end session and reset
						if (ctx_data_map[ctx].curRanges == ctx_data_map[ctx].maxNumRanges)
						{
							endSession(ctx_data_map[ctx]);
							//printContextMetrics(ctx_data_map[ctx]);
							ctx_data_map[ctx].curRanges = 0;
						}

						// If no currently enabled session on this context, start one
						if (ctx_data_map[ctx].curRanges == 0 && ctx_data_map[ctx].iterations < ctx_data_map[ctx].maxSessions)
						{

							initialize_ctx_data(ctx_data_map[ctx]);
							startSession(ctx_data_map[ctx]);

						}

						// Increment curRanges
						ctx_data_map[ctx].curRanges++;
					}
					ctx_data_mutex.unlock();			
				}
			}
		}
		else if (domain == CUPTI_CB_DOMAIN_RESOURCE){
		
			if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)// contex should be created by default on the first CUDA runtime API call
			{
				CUpti_ResourceData const * res_data = static_cast<CUpti_ResourceData const *>(cbdata);
				CUcontext ctx = res_data->context;
				ctxProfilerData data = { };
				data.ctx = ctx;
				
				try{
					RUNTIME_API_CALL(cudaGetDevice(&(data.dev_id)));

					RUNTIME_API_CALL(cudaGetDeviceProperties(&(data.dev_prop), data.dev_id));
				}
				catch(exception& e ){
					cout << e.what() << endl;
					exit(EXIT_FAILURE);
				}
				
				ctx_data_mutex.lock();
				if (data.dev_prop.major >= 7) //check compute capability 
				{
				   ctx_data_map[ctx] = data;
				   initialize_ctx_data(ctx_data_map[ctx]);
				}
				else if (ctx_data_map.count(ctx))
				{
					ctx_data_map.erase(ctx);
				}
				ctx_data_mutex.unlock();
			}
			
		  
			
		}		
	}

	void startSession(ctxProfilerData &ctx_data){
		try	
		{
			CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
			beginSessionParams.counterDataImageSize = ctx_data.counterDataImage.size();
			beginSessionParams.pCounterDataImage = ctx_data.counterDataImage.data();
			beginSessionParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();
			beginSessionParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
			beginSessionParams.ctx = ctx_data.ctx;
			beginSessionParams.maxLaunchesPerPass = ctx_data.maxNumRanges;
			beginSessionParams.maxRangesPerPass = ctx_data.maxNumRanges;
			beginSessionParams.pPriv = NULL;
			beginSessionParams.range = ctx_data.profilerRange;
			beginSessionParams.replayMode = CUPTI_KernelReplay;
			CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

			CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
			setConfigParams.pConfig = ctx_data.configImage.data();
			setConfigParams.configSize = ctx_data.configImage.size();
			setConfigParams.passIndex = 0; // Only set for Application Replay mode
			setConfigParams.minNestingLevel = 1;
			setConfigParams.numNestingLevels = 1;
			setConfigParams.targetNestingLevel = 1;
			CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

			CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
			enableProfilingParams.ctx = ctx_data.ctx;
			CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
			ctx_data.results.clear();
			ctx_data.iterations++;
		}
		catch(exception& e ){
			cout << e.what() << endl;
			exit(EXIT_FAILURE);
		}
		catch(... ){
			cout << "unknown failure" << endl;
			exit(EXIT_FAILURE);
		}	
		
	}

	void endSession(ctxProfilerData &ctx_data){
		try{
			CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
			disableProfilingParams.ctx = ctx_data.ctx;
			CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

			CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
			unsetConfigParams.ctx = ctx_data.ctx;
			CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

			CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
			endSessionParams.ctx = ctx_data.ctx;
			CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
			
			//save results
			cupMetrics.getMetricsDatafromContextData(ctx_data);

			// Clear counterDataImage 
			CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
			initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
			initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
			initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
			initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
			CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));
		}
		catch(exception& e ){
			cout << e.what() << endl;
			exit(EXIT_FAILURE);
		}
		catch(... ){
			cout << "unknown failure" << endl;
			exit(EXIT_FAILURE);
		}	
	}

	void exitCB(){

		ctx_data_mutex.lock();

		for (auto itr = ctx_data_map.begin(); itr != ctx_data_map.end(); ++itr)
		{
			ctxProfilerData &data = itr->second;

			if (data.curRanges > 0)
			{
				printContextMetrics(data);
				writeTofile("metricDump", data.results,data.dev_prop.name, *(cupMetrics.getMetricVector()));
				ctxProfilerData data2;
				readtoContext("metricDump",data2);
				data.curRanges = 0;
				
			}
		}

		ctx_data_mutex.unlock();
		
	}

	void subscribeCB(){
			//CUpti_SubscriberHandle subscriber;
			CUPTI_API_CALL(cuptiSubscribe(&cupti_subscriber, (CUpti_CallbackFunc)(callback), NULL));
			CUPTI_API_CALL(cuptiEnableCallback(1, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
			CUPTI_API_CALL(cuptiEnableCallback(1, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
		
		// Register callback for application exit
			atexit(exitCB);	
	}

	void unSubscribeCB(){
			CUPTI_API_CALL(cuptiUnsubscribe (  cupti_subscriber ));
	}

}