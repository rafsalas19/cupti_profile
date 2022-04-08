#pragma once

struct ctxProfilerData;
#include <vector>
#include <string>

bool getMetricsDatafromContextData(const ctxProfilerData &ctx,const std::vector<std::string>& metricNames);