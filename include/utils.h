#pragma once

#include <vector>
#include <string>

struct MetricRecord;
struct ctxProfilerData;

bool writeTofile(std::string pFileName, const std::vector<MetricRecord>& metricRecords,std::string chipname, const std::vector<std::string>& metricNames );

bool readtoContext( std::string pFileName, ctxProfilerData &ctx);