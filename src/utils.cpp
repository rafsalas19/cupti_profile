#include<iostream>
#include<fstream>
#include "../include/utils.h"
#include "../include/cuptiMetrics.h"
#include "../include/profileSession.h"
#include <stdlib.h>
#include <iomanip>
using namespace std;





bool writeTofile(std::string pFileName, const std::vector<MetricRecord>& metricRecords,string chipname, const std::vector<std::string>& metricNames ){
	try{
		ofstream metricFile( pFileName, ios::out | ios::binary |ios::trunc);
		if(!metricFile) {
			throw std::runtime_error( "Cannot open file!");
		}
		
		size_t size =chipname.size();
		
		metricFile.write((char*)&size,sizeof(size));
		metricFile.write((char*)&chipname[0],size);
		

		string mnames;
		for(auto metricName : metricNames){
			mnames=mnames+ metricName +":";
		}
		
		size =mnames.size();
		metricFile.write((char*)&size,sizeof(size));
		metricFile.write((char*)&mnames[0],size);
		
		size =metricRecords.size();
		metricFile.write((char*)&size,sizeof(size));
		
		for(auto record :metricRecords){
			string serialStr;
			record.serialize(size,serialStr);
			metricFile.write((char*)&size,sizeof(size));
			metricFile.write((char*)&serialStr[0],size);
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
		fstream metricFile( pFileName, ios::in | ios::binary);
		if(!metricFile) {
			throw std::runtime_error( "Cannot open file!");
		}
		//auto file_size = std::filesystem::file_size(pFileName);
		
		string chipname;
		size_t size;
		
		metricFile.read((char*)&size,sizeof(size));
		chipname.resize(size);
		metricFile.read((char*)&chipname[0],size);
		
		//metric names
		string mnames;
		metricFile.read((char*)&size,sizeof(size));
		mnames.resize(size);
		metricFile.read((char*)&mnames[0],size);

		metricFile.read((char*)&size,sizeof(size));
		MetricRecord mr;
		for(auto i=0;i<size;++i){
			size_t strSize;
			string serialStr;
			metricFile.read((char*)&strSize,sizeof(strSize));
			serialStr.resize(strSize);
			metricFile.read((char*)&serialStr[0],strSize);
			ctx.push_back(mr);
		}
		
		metricFile.close();
	}
	catch(exception &e){
		cout<< e.what()<<endl;
		return false;
	}
	return true;
}