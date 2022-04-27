#include <iostream>
#include "../include/cuptiMetrics.h"
#include "../include/profileSession.h"
#include "../include/cuptiErrorCheck.h"
#include "../include/PWMetrics.h"
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

using namespace std;


MetricRecord::MetricRecord(string _rangeName,string _metricName, double _metricValue): rangeName(_rangeName),metricName(_metricName),metricValue(_metricValue){}
MetricRecord::MetricRecord(){}

void MetricRecord::printRecord(int w1,int w2){
	cout << setw(w1) << left << rangeName << setw(w2)
	<< left << metricName << metricValue << endl;		
}
void MetricRecord::serialize(size_t &size, string &serialStr){
	serialStr=rangeName+":"+metricName+":"+to_string(metricValue);
	size=serialStr.size();
}
void MetricRecord::deserialize(string &str){
	string tmpDbl;		
	int end =str.find(':');
	rangeName=str.substr(0, end);
	str=str.substr(end+1,str.size());
	end =str.find(':');
	metricName=str.substr(0, end);
	tmpDbl=str.substr(end+1,str.size());
	metricValue= stod(tmpDbl);
	//cout<<rangeName<<" "<<metricName<<" "<< tmpDbl<<endl;
}



CuptiMetrics::CuptiMetrics():metricToPWFormula(&Metrics::_metricToPWFormula),metricCodeMap(&Metrics::_metricCodeMap){

}

bool CuptiMetrics::validMetric(string metric){
	if ( metricToPWFormula->find(metric) == metricToPWFormula->end() ) {
		return false;
	} 
	else {
		return true;
	}
}

string CuptiMetrics::getFormula(int metricCode){
	if ( metricCodeMap->find(metricCode) != metricCodeMap->end() ){
		return getFormula((*metricCodeMap)[metricCode]);
	} 
	else {
		throw std::runtime_error("metric code "+ to_string(metricCode) +" not found"); 
	}
}

string CuptiMetrics::getFormula(string metric){
	if ( metricToPWFormula->find(metric) != metricToPWFormula->end() ){
		return (*metricToPWFormula)[metric];
	} 
	else {
		throw std::runtime_error("metric not found"); 
	}
}

	//need to add error checking
bool CuptiMetrics::getMetricRequests(ctxProfilerData &ctx_data, vector<NVPA_RawMetricRequest> &rawMetricRequests){
		//vector<NVPA_RawMetricRequest> rawMetricRequests;
		auto chipName=ctx_data.dev_prop.name;
		auto cntrAvailImg=ctx_data.counterAvailabilityImage.data();
		
		NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calcScratchBuffSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        calcScratchBuffSizeParam.pChipName = chipName;
        calcScratchBuffSizeParam.pCounterAvailabilityImage =cntrAvailImg; 
		NVPW_ERROR_CHECK(  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calcScratchBuffSizeParam));
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
			validMetric(metricName);
					
			NVPW_MetricEvalRequest metricEvalReq;
			NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalReq = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
			convertMetricToEvalReq.pMetricsEvaluator = metricEvaluator;
			convertMetricToEvalReq.pMetricName = metricName.c_str();
			convertMetricToEvalReq.pMetricEvalRequest = &metricEvalReq;
			convertMetricToEvalReq.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
			NVPW_ERROR_CHECK( NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalReq));

			std::vector<const char*> rawDependencies;
			//get metric dependencies 
			NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDepParms = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
			getMetricRawDepParms.pMetricsEvaluator = metricEvaluator;
			getMetricRawDepParms.pMetricEvalRequests = &metricEvalReq;
			getMetricRawDepParms.numMetricEvalRequests = 1;
			getMetricRawDepParms.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
			getMetricRawDepParms.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
			NVPW_ERROR_CHECK( NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDepParms));	
			rawDependencies.resize(getMetricRawDepParms.numRawDependencies);
			getMetricRawDepParms.ppRawDependencies = rawDependencies.data();
			NVPW_ERROR_CHECK( NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDepParms));
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
        NVPW_ERROR_CHECK( NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
		return true;		
	}
	
bool CuptiMetrics::configureConfigImage(ctxProfilerData &ctx_data){
		try{
			vector<NVPA_RawMetricRequest> rawMetricRequests;
			if(!getMetricRequests(ctx_data,rawMetricRequests)){
				//need error checking
				return false;
			}
			auto chipName=ctx_data.dev_prop.name;
			auto cntrAvailImg=ctx_data.counterAvailabilityImage.data();
			NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
			rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
			rawMetricsConfigCreateParams.pChipName = chipName;
			rawMetricsConfigCreateParams.pCounterAvailabilityImage = cntrAvailImg;
			NVPW_ERROR_CHECK( NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams));
			NVPA_RawMetricsConfig* pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;
			
			if(cntrAvailImg)
			{
				NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
				setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
				setCounterAvailabilityParams.pCounterAvailabilityImage = cntrAvailImg;
				NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams));
			}
			
			NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGrpParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
			beginPassGrpParams.pRawMetricsConfig = pRawMetricsConfig;
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGrpParams));		
			
			NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
			addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
			addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
			addMetricsParams.numMetricRequests = rawMetricRequests.size();
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));
			
			NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
			endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));
			
			NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfImgParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
			generateConfImgParams.pRawMetricsConfig = pRawMetricsConfig;
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfImgParams));

			NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
			getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
			getConfigImageParams.bytesAllocated = 0;
			getConfigImageParams.pBuffer = NULL;
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

			ctx_data.configImage.resize(getConfigImageParams.bytesCopied);
			getConfigImageParams.bytesAllocated = ctx_data.configImage.size();
			getConfigImageParams.pBuffer = ctx_data.configImage.data();
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));		
			
			//counter data prefix set up
			NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE };
			counterDataBuilderCreateParams.pChipName = chipName;
			counterDataBuilderCreateParams.pCounterAvailabilityImage = cntrAvailImg;
			NVPW_ERROR_CHECK( NVPW_CUDA_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

			NVPW_CounterDataBuilder_AddMetrics_Params cbd_addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
			cbd_addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
			cbd_addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
			cbd_addMetricsParams.numMetricRequests = rawMetricRequests.size();
			NVPW_ERROR_CHECK( NVPW_CounterDataBuilder_AddMetrics(&cbd_addMetricsParams));

			size_t counterDataPrefixSize = 0;
			NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
			getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
			getCounterDataPrefixParams.bytesAllocated = 0;
			getCounterDataPrefixParams.pBuffer = NULL;
			NVPW_ERROR_CHECK( NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

			ctx_data.counterDataPrefixImage.resize(getCounterDataPrefixParams.bytesCopied);
			getCounterDataPrefixParams.bytesAllocated = ctx_data.counterDataPrefixImage.size();
			getCounterDataPrefixParams.pBuffer = ctx_data.counterDataPrefixImage.data();
			NVPW_ERROR_CHECK( NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));
					
			//cleanup	
			NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
			counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
			NVPW_ERROR_CHECK( NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams));
					
			NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
			rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
			NVPW_ERROR_CHECK( NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams));
		
			return true;
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

bool CuptiMetrics::getMetricsDatafromContextData( ctxProfilerData &ctx){
	try{
		if (!ctx.counterDataImage.size())
		{
			cout << "Counter Data Image is empty!\n";
			return false;
		}
		NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
		calculateScratchBufferSizeParam.pChipName = ctx.dev_prop.name;
		calculateScratchBufferSizeParam.pCounterAvailabilityImage = ctx.counterAvailabilityImage.data();
		NVPW_ERROR_CHECK(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

		vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
		NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
		metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
		metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
		metricEvaluatorInitializeParams.pChipName = ctx.dev_prop.name;
		metricEvaluatorInitializeParams.pCounterAvailabilityImage = ctx.counterAvailabilityImage.data();
		metricEvaluatorInitializeParams.pCounterDataImage = ctx.counterDataImage.data();
		metricEvaluatorInitializeParams.counterDataImageSize = ctx.counterDataImage.size();
		NVPW_ERROR_CHECK(NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
		NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;
		
		NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
		getNumRangesParams.pCounterDataImage = ctx.counterDataImage.data();
		NVPW_ERROR_CHECK(NVPW_CounterData_GetNumRanges(&getNumRangesParams));
		
		for (string metricName : metricNames)
		{
			NVPW_MetricEvalRequest metricEvalRequest;
			NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
			convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
			convertMetricToEvalRequest.pMetricName = metricName.c_str();
			convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
			convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
			NVPW_ERROR_CHECK(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

			for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
			{
				NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
				getRangeDescParams.pCounterDataImage = ctx.counterDataImage.data();
				getRangeDescParams.rangeIndex = rangeIndex;
				NVPW_ERROR_CHECK(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
				vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
				getRangeDescParams.ppDescriptions = descriptionPtrs.data();
				NVPW_ERROR_CHECK( NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

				string rangeName;
				for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
				{
					if (descriptionIndex)
					{
						rangeName += "/";
					}
					rangeName += descriptionPtrs[descriptionIndex];
				}
				
				NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
				setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
				setDeviceAttribParams.pCounterDataImage = ctx.counterDataImage.data();
				setDeviceAttribParams.counterDataImageSize = ctx.counterDataImage.size();
				NVPW_ERROR_CHECK(NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

				double metricValue;
				NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
				evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
				evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
				evaluateToGpuValuesParams.numMetricEvalRequests = 1;
				evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
				evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
				evaluateToGpuValuesParams.pCounterDataImage = ctx.counterDataImage.data();
				evaluateToGpuValuesParams.counterDataImageSize = ctx.counterDataImage.size();
				evaluateToGpuValuesParams.rangeIndex = rangeIndex;
				evaluateToGpuValuesParams.isolated = true;
				evaluateToGpuValuesParams.pMetricValues = &metricValue;
				NVPW_ERROR_CHECK(NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));
				
				MetricRecord metricRecord(rangeName,metricName,metricValue);
				ctx.results.push_back(metricRecord);
			}
		}
		
		NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
		metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
		NVPW_ERROR_CHECK(NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
	}
	catch(exception& e ){
		cout << e.what() << endl;
		exit(EXIT_FAILURE);
	}
	catch(... ){
		cout << "unknown failure" << endl;
		exit(EXIT_FAILURE);
	}
	return true;
}


void CuptiMetrics::printMetricRecords(const ctxProfilerData &ctx){
	int w1=40;
	int w2 = 100;
	cout << "\n" << setw(w1) << left << "Range Name"
			  << setw(w2) << left        << "Metric Name"
			  << "Metric Value" << endl;
	cout << setfill('-') << setw(160) << "" << setfill(' ') << endl;
	
	for(auto mRecord : ctx.results){
		mRecord.printRecord(w1,w2);
	}
}