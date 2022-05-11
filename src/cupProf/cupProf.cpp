
#include "dlfcn.h" // dlsym, RTLD_NEXT
#include <vector>
#include <string>
#include "cupti_driver_cbid.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "../include/cuptiErrorCheck.h"
#include "../include/cupProf/cupProf.h"
#include "../include/cuptiMetrics.h"
#include "../include/PWMetrics.h"

using namespace std;

void deserialize_codes(string &str,vector<int> &tokens){
	while(str.size()> 1){
		
		int end =str.find(",");
		if(end<3){
			end=str.size();
		}
		string tmp =str.substr(0, end);
		int code;
		try{
			code=stoi(tmp);	
			if(CODE_MAP_MIN_KEY > code || CODE_MAP_MAX_KEY < code){
				throw std::runtime_error("Code is out of bounds.");
			}
			tokens.push_back(code);
		}
        catch (const std::invalid_argument & e) {
            std::cout << e.what() << ". Invalid code format "+tmp +". Skipping. \n";
			
        }
        catch (const std::out_of_range & e) {
            std::cout << e.what() << ". Invalid code format "+tmp +" Skipping. \n";
        }
		catch(exception& e ){
			 std::cout << e.what() << " Skipping. \n";
		}
		
		if(end==str.size()) break;
		str=str.substr(end+1,str.size());

	}
}


cupProfiler::cupProfiler(){
	try{
		//int numDevices;
		//RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));
		//cout<<"Device count: "<<numDevices <<endl;
		
		//set Metrics
		char * env_var = getenv("CUPTI_METRIC_CODES");
		if(env_var!=NULL){
			string str =env_var;
			vector<int> metricCodes;
			deserialize_codes(str,metricCodes);
			
			for (auto& code : metricCodes){
				try{
					cupMetrics.getMetricVector()->push_back(cupMetrics.getFormula(code));
				}
				catch(exception& e ){
					std::cout << e.what() << " Skipping. \n";
					continue;
				}			
			}
		}
		if(env_var==NULL ||cupMetrics.getMetricVector()->size()==0){
			cout<< "Metrics not provided to env variable CUPTI_METRIC_CODES, running default metric collection"<<endl;

			cupMetrics.getMetricVector()->push_back(cupMetrics.getFormula(1000));//achieved_occupancy
			cupMetrics.getMetricVector()->push_back(cupMetrics.getFormula(1005));//dram_read_throughput
			cupMetrics.getMetricVector()->push_back(cupMetrics.getFormula(1018));//flop_count_sp_add
			
				//{1, "sm__warps_active.avg.pct_of_peak_sustained_active"},
        		//{2, "dram__bytes_read.sum.per_second"},
        		//{3, "smsp__thread_inst_executed_per_inst_executed.ratio"},
						
		}		
			
	}
	catch(exception& e ){
		cout << e.what() << endl;
		exit(EXIT_FAILURE);
	}
			
			

	
}
cupProfiler::~cupProfiler(){}


void cupProfiler::initializeContextData(CUcontext &ctx){
	//check does context exist
	//to do
	int device;
	struct cudaDeviceProp props;
	RUNTIME_API_CALL(cudaGetDevice(&device));
	RUNTIME_API_CALL(cudaGetDeviceProperties(&props, device));
	if(props.major < 7){
		cout<<"Cuda Compute capability less than 7.0 not supported"<<endl;
		exit(EXIT_FAILURE);
	}
	ctxProfilerData data = { };
	data.ctx = ctx;
	ctx_mutex.lock();
	ctx_data_map[ctx] = data;

	ctx_mutex.unlock();
	ProfileSession::initialize_ctx_data(ctx_data_map[ctx]);
	
	
}


void cupProfiler::cupCB(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata){
	
	
}

void cupProfiler::cupSubscribe(){
	CUPTI_API_CALL(cuptiSubscribe(&cupSubscriber, (CUpti_CallbackFunc)(cupCB), NULL));
	CUPTI_API_CALL(cuptiEnableCallback(1, cupSubscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
	CUPTI_API_CALL(cuptiEnableCallback(1, cupSubscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));		
}		


			
void cupProfiler::startSession(){
	cudaDeviceSynchronize();// this is needed to ensure the CUDA context is fully initialized. Needed due to lazy cuda initialization
	CUcontext ctx;
	DRIVER_API_CALL(cuCtxGetCurrent(&ctx));
	
	if(ctx_data_map.find(ctx) == ctx_data_map.end()){
		initializeContextData(ctx);
	}


	ProfileSession::startSession(ctx_data_map[ctx]);
	cout<<"start"<<endl;
}
void cupProfiler::endSession(){
	CUcontext ctx;
	DRIVER_API_CALL(cuCtxGetCurrent(&ctx));
	initializeContextData(ctx);
	ProfileSession::endSession(ctx_data_map[ctx]);
	cout<<"end"<<endl;
}			