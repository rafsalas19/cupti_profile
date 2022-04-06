#include <iostream>
#include "../include/cuptiMetrics.h"
#include "../include/profileSession.h"
//NVPA_STATUS_SUCCESS

//mirroring nvidia cupti sample extension. Specificly Metrics portion

namespace Metrics{
	
	metricList get_metricList(){
		metricList metrics=
		{
		        {1, "sm__warps_active.avg.pct_of_peak_sustained_active"},
        		{2, "dram__bytes_read.sum.per_second"},
        		{3, "smsp__thread_inst_executed_per_inst_executed.ratio"},
		
		};
		
		return metrics;
	}
	
	bool metricCheck(const string& metricName){
		//code to make sure that the requested metric is acceptable
		 
			return true;
	}
	//need to add error checking
	bool getMetricRequests(ctxProfilerData &ctx_data,const std::vector<std::string>& metricNames,vector<NVPA_RawMetricRequest> &rawMetricRequests){
		//vector<NVPA_RawMetricRequest> rawMetricRequests;
		auto chipName=ctx_data.dev_prop.name;
		auto cntrAvailImg=ctx_data.counterAvailabilityImage.data();
		
		NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calcScratchBuffSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        calcScratchBuffSizeParam.pChipName = chipName;
        calcScratchBuffSizeParam.pCounterAvailabilityImage =cntrAvailImg; 
		NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calcScratchBuffSizeParam);
		//make evaluator
		vector<uint8_t> scratchBuff(calcScratchBuffSizeParam.scratchBufferSize);
		NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvalInitParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
		metricEvalInitParams.scratchBufferSize = scratchBuff.size();
        metricEvalInitParams.pScratchBuffer = scratchBuff.data();
        metricEvalInitParams.pChipName = chipName;
		metricEvalInitParams.pCounterAvailabilityImage = cntrAvailImg;
        NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvalInitParams);	
        NVPW_MetricsEvaluator* metricEvaluator = metricEvalInitParams.pMetricsEvaluator;
		
		//convert metric into request part 1
		std::vector<const char*> rawMetricNames;
		for (auto& metricName : metricNames)
		{
			//check if valid metric
			metricCheck(metricName);
					
			NVPW_MetricEvalRequest metricEvalReq;
			NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalReq = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
			convertMetricToEvalReq.pMetricsEvaluator = metricEvaluator;
			convertMetricToEvalReq.pMetricName = metricName.c_str();
			convertMetricToEvalReq.pMetricEvalRequest = &metricEvalReq;
			convertMetricToEvalReq.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
			NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalReq);

			std::vector<const char*> rawDependencies;
			//get metric dependencies 
			NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDepParms = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
			getMetricRawDepParms.pMetricsEvaluator = metricEvaluator;
			getMetricRawDepParms.pMetricEvalRequests = &metricEvalReq;
			getMetricRawDepParms.numMetricEvalRequests = 1;
			getMetricRawDepParms.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
			getMetricRawDepParms.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
			NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDepParms);	
			rawDependencies.resize(getMetricRawDepParms.numRawDependencies);
			getMetricRawDepParms.ppRawDependencies = rawDependencies.data();
			NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDepParms);
			for (size_t i = 0; i < rawDependencies.size(); ++i)
			{
				rawMetricNames.push_back(rawDependencies[i]);
			}
		}
		//convert metric into request part 2		
		for (auto& rawMetricName : rawMetricNames)
		{
			NVPA_RawMetricRequest metricReq = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
			metricReq.pMetricName = rawMetricName;
			metricReq.isolated = true;
			metricReq.keepInstances = true;
			rawMetricRequests.push_back(metricReq);
		}
		//clean up
		NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
        metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
        NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams);
		return true;		
	}
	
	bool configureConfigImage(ctxProfilerData &ctx_data,const vector<std::string>& metricNames){
		vector<NVPA_RawMetricRequest> rawMetricRequests;
		if(!getMetricRequests(ctx_data,metricNames,rawMetricRequests)){
			//need error checking
			return false;
		}
		auto chipName=ctx_data.dev_prop.name;
		auto cntrAvailImg=ctx_data.counterAvailabilityImage.data();
		NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
        rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        rawMetricsConfigCreateParams.pChipName = chipName;
		rawMetricsConfigCreateParams.pCounterAvailabilityImage = cntrAvailImg;
        NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams);
        NVPA_RawMetricsConfig* pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;
		
		if(cntrAvailImg)
        {
			NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
			setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
			setCounterAvailabilityParams.pCounterAvailabilityImage = cntrAvailImg;
			NVPA_Status status = NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams);
			if (NVPA_STATUS_SUCCESS != status) return false;
		}
		
		NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGrpParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
		beginPassGrpParams.pRawMetricsConfig = pRawMetricsConfig;
		NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGrpParams);		
		
		NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
		addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
		addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
		addMetricsParams.numMetricRequests = rawMetricRequests.size();
		NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams);
		
		NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
        endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
        NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams);
		
		NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfImgParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
		generateConfImgParams.pRawMetricsConfig = pRawMetricsConfig;
		NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfImgParams);

		NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
		getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
		getConfigImageParams.bytesAllocated = 0;
		getConfigImageParams.pBuffer = NULL;
		NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams);

		ctx_data.configImage.resize(getConfigImageParams.bytesCopied);
		getConfigImageParams.bytesAllocated = ctx_data.configImage.size();
		getConfigImageParams.pBuffer = ctx_data.configImage.data();
		NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams);
		
		
		
		//counter data prefix set up
		NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE };
		counterDataBuilderCreateParams.pChipName = chipName;
		counterDataBuilderCreateParams.pCounterAvailabilityImage = cntrAvailImg;
		NVPW_CUDA_CounterDataBuilder_Create(&counterDataBuilderCreateParams);

		NVPW_CounterDataBuilder_AddMetrics_Params cbd_addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
		cbd_addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
		cbd_addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
		cbd_addMetricsParams.numMetricRequests = rawMetricRequests.size();
		NVPW_CounterDataBuilder_AddMetrics(&cbd_addMetricsParams);

		size_t counterDataPrefixSize = 0;
		NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
		getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
		getCounterDataPrefixParams.bytesAllocated = 0;
		getCounterDataPrefixParams.pBuffer = NULL;
		NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams);

		ctx_data.counterDataPrefixImage.resize(getCounterDataPrefixParams.bytesCopied);
		getCounterDataPrefixParams.bytesAllocated = ctx_data.counterDataPrefixImage.size();
		getCounterDataPrefixParams.pBuffer = ctx_data.counterDataPrefixImage.data();
		NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams);
				
		//cleanup	
		NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
		counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
		NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams);
				
		NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
        rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
		NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams);
	}
}