#pragma once

#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

struct ctxProfilerData;



struct MetricRecord{
	MetricRecord(string _rangeName,string _metricName, string _metricValue){}
	MetricRecord(){}
	string rangeName;
	string metricName; 
	string metricValue;
};

class CuptiMetrics{
public:
	CuptiMetrics();
	
	bool validMetric(string metric);
	string getFormula(int metricCode);
	string getFormula(string metric);
	
	//create counter Availability Image + counter data prefix image
	bool configureConfigImage(ctxProfilerData &ctx_data);
	//gather the metric values into a record
	bool getMetricsDatafromContextData(const ctxProfilerData &ctx);
	
	vector<string> * getMetricVector(){return &metricNames;}
	
private:
	//create raw metric request for perf works
	bool getMetricRequests(ctxProfilerData &ctx_data, vector<NVPA_RawMetricRequest> &rawMetricRequests);
	unordered_map<string, string> *metricToPWFormula;
	unordered_map<unsigned int, string> *metricCodeMap;	
	vector<string> metricNames;
	vector <MetricRecord> results; 

};

