#pragma once

#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;

struct ctxProfilerData;

namespace Metrics{
	typedef std::map<int, string> metricList;

	metricList get_metricList();
	
	bool getMetricRequests(ctxProfilerData &ctx_data,const vector<std::string>& metricNames,vector<NVPA_RawMetricRequest> &rawMetricRequests);
	
	bool configureConfigImage(ctxProfilerData &ctx_data,const vector<std::string>& metricNames);
}

