#include<iostream>
#include<fstream>
#include <stdlib.h>
#include <iomanip>
#include <getopt.h>

#include "cuptiMetrics.h"
#include "profileSession.h"
#include "utils.h"
#include "PWMetrics.h"

using namespace std;

void deserialize(string &str,string delimiter,vector<string> &tokens){
	while(str.size()> 1){		
		int end =str.find(delimiter);
		tokens.push_back(str.substr(0, end));
		str=str.substr(end+1,str.size());
	}
}

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
			ctx.results.push_back(mr);
		}
		
		metricFile.close();
	}
	catch(exception &e){
		cout<< e.what()<<endl;
		return false;
	}
	return true;
}

void printHelp(){
	std::cout << "Usage:" << std::endl;
	std::cout << "\t--output_file or -o <file_path>." << std::endl;
	std::cout << "\t--metric_field_codes or -c. Provide comma separated field codes." << std::endl;
	std::cout << "\t--list_metrics or -l. Provides list of field code to metric mapping." << std::endl;
	std::cout << "\t--number_kernel_launches or -k. Number of kernel launches before ending the session." << std::endl;
	exit(0);
	
}

void get_opts(int argc, char **argv, struct options_t *opts)
{
    if (argc == 1)
    {
		printHelp();
    }


    struct option l_opts[] = {
        {"output_file", no_argument, NULL, 'o'},
        {"metric_field_codes", no_argument, NULL, 'c'},
        {"list_metrics", no_argument, NULL, 'l'},
        {"number_kernel_launches", no_argument, NULL, 'k'},
		{"help", no_argument, NULL, 'h'},
		{NULL,0, 0, 0}
	
    };

    int ind, c;
	opts->list_metrics =false;
	opts->number_kernel_launches=1;
	opts->out_file=NULL;
	opts->field_codes=NULL;
	cout<<argv[1]<<endl;
	while ((c = getopt_long(argc, argv, ":o:c:k:lh", l_opts, &ind)) != -1)
    {
        switch (c)
        {
			case 0:		
				break;
			case 'o':
				opts->out_file = (char *)optarg;
				break;
			case 'c'://check if we need to loop
				opts->field_codes = (char *)optarg; 
				break;
			case 'l':
				opts->list_metrics = true;//std::stod((char *)optarg);
				break;                                    
			case 'k':
				opts->number_kernel_launches = atoi((char *)optarg);
				break;
			case 'h':
				printHelp();
				break;
			case ':':
				std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
				exit(1);
			
        }
    }
	// for(; ind < argc; ind++){ //when some extra arguments are passed
      // printf("Given extra arguments: %s\n", argv[ind]);
    // }
}


void printMetrics(){
	int w1=40;
	int w2 = 100;
	cout << "\n" << setw(w1) << left << "Range Name"<< setw(w2) << left << "Metric Name"<< endl;
	cout << setfill('-') << setw(160) << "" << setfill(' ') << endl;
	for (auto itr = Metrics::_metricCodeMap.begin(); itr != Metrics::_metricCodeMap.end(); ++itr){
		cout << setw(w1) << left << itr->first << setw(w2)<< left << itr->second <<  endl;
	}
	
	
}


