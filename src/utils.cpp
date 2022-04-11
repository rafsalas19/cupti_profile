#include<iostream>
#include<fstream>
#include "../include/utils.h"
#include "../include/cuptiMetrics.h"
#include <stdlib.h>
#include <iomanip>
using namespace std;

bool writeTofile(std::string pFileName, const std::vector<MetricRecord>& metricRecords,string chipname, const std::vector<std::string>& metricNames ){
	try{
		ofstream metricFile( pFileName, ios::out | ios::binary |ios::trunc);
		if(!metricFile) {
			throw std::runtime_error( "Cannot open file!");
		}
		metricFile.write((char*)&chipname,sizeof(string));
		int size =metricNames.size();
		metricFile.write((char*)&size,sizeof(int));
		
		for(auto metricName : metricNames){
			metricFile.write((char*)&metricName,sizeof(string));
		}
		
		size =metricRecords.size();
		metricFile.write((char*)&size,sizeof(int));
		
		for(auto record : metricRecords){
			metricFile.write((char*)&record,sizeof(MetricRecord));
		}
		metricFile.close();
	}
	catch(exception &e){
		cout<< e.what()<<endl;
		return false;
	}
	return true;
}

bool readtoContext( std::string pFileName,ctxProfilerData &ctx){
	try{
		ifstream metricFile( pFileName, ios::in | ios::binary);
		if(!metricFile) {
			throw std::runtime_error( "Cannot open file!");
		}
		string chipname;
		metricFile.read((char*)&chipname,sizeof(string));
		cout<<chipname<<endl;
		/*metricFile.write((char*)&metricNames.size(),sizeof(int));
		for(auto metricName : metricNames){
			metricFile.write((char*)&metricName,sizeof(string));
		}
		metricFile.write((char*)&metricRecords.size(),sizeof(int));
		for(auto record : metricRecords){
			metricFile.write((char*)&record,sizeof(MetricRecord));
		}*/
		metricFile.close();
	}
	catch(exception &e){
		cout<< e.what()<<endl;
		return false;
	}
	return true;
}