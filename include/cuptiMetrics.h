#pragma once

#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>
using namespace std;

struct ctxProfilerData;



struct MetricRecord{
	MetricRecord(string _rangeName,string _metricName, double _metricValue): rangeName(_rangeName),metricName(_metricName),metricValue(_metricValue){}
	MetricRecord(){}
	void printRecord(int w1=40,int w2=100){
		cout << setw(w1) << left << rangeName << setw(w2)
		<< left << metricName << metricValue << endl;		
	}
	string rangeName;
	string metricName; 
	double metricValue;
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
	bool getMetricsDatafromContextData( ctxProfilerData &ctx);
	
	vector<string> * getMetricVector(){return &metricNames;}
	void printMetricRecords(const ctxProfilerData &ctx);
	
private:
	//create raw metric request for perf works
	bool getMetricRequests(ctxProfilerData &ctx_data, vector<NVPA_RawMetricRequest> &rawMetricRequests);
	unordered_map<string, string> *metricToPWFormula;
	unordered_map<unsigned int, string> *metricCodeMap;	
	vector<string> metricNames;
	 

};

