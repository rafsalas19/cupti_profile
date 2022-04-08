#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <iostream>
#include "../include/utils.h"
#include "../include/cuptiErrorCheck.h"
#include "../include/profileSession.h"
#include "../include/PWMetrics.h"
#include <iomanip>
using namespace std;

bool getMetricsDatafromContextData(const ctxProfilerData &ctx,const std::vector<std::string>& metricNames){
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

		cout << "\n" << setw(40) << left << "Range Name"
				  << setw(100) << left        << "Metric Name"
				  << "Metric Value" << endl;
		cout << setfill('-') << setw(160) << "" << setfill(' ') << endl;

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
				
				cout << setw(40) << left << rangeName << setw(100)
						  << left << metricName << metricValue << endl;
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
	
