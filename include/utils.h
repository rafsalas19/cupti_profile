#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

struct MetricRecord;
struct ctxProfilerData;

bool writeTofile(std::string pFileName, const std::vector<MetricRecord>& metricRecords,std::string chipname, const std::vector<std::string>& metricNames );

bool readtoContext( std::string pFileName, ctxProfilerData &ctx);

struct options_t {
    char *out_file;
	char *field_codes;
	bool list_metrics;
	int number_kernel_launches;
	
};

void get_opts(int argc, char **argv, struct options_t *opts);

void deserialize(std::string &str,std::string delimiter,std::vector<std::string> &tokens);



void printMetrics();

#endif