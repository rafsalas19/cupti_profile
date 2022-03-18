#pragma once

#include <map>
#include <stdlib.h>

namespace Metrics{
	typedef std::map<string, int> metricList;

	metricList get_metricList(){
		metricList metrics=
		{
		        {1, "sm__warps_active.avg.pct_of_peak_sustained_active"},
        		{2, "dram__bytes_read.sum.per_second"},
        		{3, "smsp__thread_inst_executed_per_inst_executed.ratio"}
		
		}


		return metricList;
	}
}

