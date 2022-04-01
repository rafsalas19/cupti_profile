#pragma once

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

using ::std::pair
struct ctxProfilerData;

class ProfileSession{

public:
	ProfileSession();
	~ProfileSession(){}
	void startSession();

	void endSession();
	
	void subscribeCB();
	
	void addMetric(std::string metric){ metricNames.push_back(metric);}

private:

	std::vector<std::string> metricNames;
	
	static void callback(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata);

	static void exitCB();
	
	mutex ctx_data_mutex;
	
	unordered_map<CUcontext, ctxProfilerData> ctx_data;

	
};
