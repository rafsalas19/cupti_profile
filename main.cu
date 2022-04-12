#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <vector>
#include <cuda.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "cuptiErrorCheck.h"
#include "cuptiMetrics.h"
#include <getopt.h>
#include <iomanip>
#include "../include/PWMetrics.h"

using namespace std;
string slist[] {"achieved_occupancy",
"branch_efficiency",
"cf_executed",
"double_precision_fu_utilization",
"dram_read_bytes",
"dram_read_throughput",
"dram_read_transactions",
"dram_utilization",
"dram_write_bytes",
"dram_write_throughput",
"dram_write_transactions",
"eligible_warps_per_cycle",
"flop_count_dp_add",
"flop_count_dp_fma",
"flop_count_dp_mul",
"flop_count_hp_add",
"flop_count_hp_fma",
"flop_count_hp_mul",
"flop_count_sp_add",
"flop_count_sp_fma",
"flop_count_sp_mul",
"flop_dp_efficiency",
"flop_hp_efficiency",
"flop_sp_efficiency",
"gld_efficiency",
"gld_throughput",
"gld_transactions",
"gld_transactions_per_request",
"global_atomic_requests",
"global_load_requests",
"global_reduction_requests",
"global_store_requests",
"gst_efficiency",
"gst_throughput",
"gst_transactions",
"gst_transactions_per_request",
"half_precision_fu_utilization",
"inst_bit_convert",
"inst_compute_ld_st",
"inst_control",
"inst_executed",
"inst_executed_global_atomics",
"inst_executed_global_loads",
"inst_executed_global_reductions",
"inst_executed_global_stores",
"inst_executed_local_loads",
"inst_executed_local_stores",
"inst_executed_shared_loads",
"inst_executed_shared_stores",
"inst_executed_surface_atomics",
"inst_executed_surface_reductions",
"inst_executed_surface_stores",
"inst_executed_tex_ops",
"inst_fp_16",
"inst_fp_32",
"inst_fp_64",
"inst_integer",
"inst_inter_thread_communication",
"inst_issued",
"inst_misc",
"inst_per_warp",
"ipc",
"issue_slot_utilization",
"issue_slots",
"issued_ipc",
"l1_sm_lg_utilization",
"l2_atomic_throughput",
"l2_atomic_transactions",
"l2_global_atomic_store_bytes",
"l2_global_load_bytes",
"l2_local_load_bytes",
"l2_read_throughput",
"l2_read_transactions",
"l2_surface_load_bytes",
"l2_surface_store_bytes",
"l2_tex_hit_rate",
"l2_tex_read_hit_rate",
"l2_tex_read_throughput",
"l2_tex_read_transactions",
"l2_tex_write_hit_rate",
"l2_tex_write_throughput",
"l2_tex_write_transactions",
"l2_utilization",
"l2_write_throughput",
"l2_write_transactions",
"ldst_fu_utilization",
"local_load_requests",
"local_load_throughput",
"local_load_transactions",
"local_load_transactions_per_request",
"local_store_requests",
"local_store_throughput",
"local_store_transactions",
"local_store_transactions_per_request",
"pcie_total_data_received",
"pcie_total_data_transmitted",
"shared_efficiency",
"shared_load_throughput",
"shared_load_transactions",
"shared_store_throughput",
"shared_store_transactions",
"shared_utilization",
"single_precision_fu_utilization",
"sm_efficiency",
"sm_tex_utilization",
"special_fu_utilization",
"stall_constant_memory_dependency",
"stall_inst_fetch",
"stall_memory_dependency",
"stall_not_selected",
"stall_sleeping",
"stall_texture",
"surface_atomic_requests",
"surface_load_requests",
"surface_reduction_requests",
"surface_store_requests",
"sysmem_read_bytes",
"sysmem_read_throughput",
"sysmem_read_transactions",
"sysmem_write_bytes",
"sysmem_write_throughput",
"sysmem_write_transactions",
"tensor_precision_fu_utilization",
"tex_cache_hit_rate",
"tex_fu_utilization",
"tex_sm_tex_utilization",
"tex_sm_utilization",
"texture_load_requests",
"warp_execution_efficiency",
"warp_nonpred_execution_efficiency"};





struct options_t {
    char *out_file;
	char *field_codes;
	bool list_metrics;
	int number_kernel_launches;
	
};

void get_opts(int argc, char **argv, struct options_t *opts);

void printMetrics(){
	int w1=40;
	int w2 = 100;
	cout << "\n" << setw(w1) << left << "Range Name"<< setw(w2) << left << "Metric Name"<< endl;
	cout << setfill('-') << setw(160) << "" << setfill(' ') << endl;
	for (auto itr = Metrics::_metricCodeMap.begin(); itr != Metrics::_metricCodeMap.end(); ++itr){
		cout << setw(w1) << left << itr->first << setw(w2)<< left << itr->second <<  endl;
	}
	
	
}

int main(int argc, char* argv[])
{
	struct options_t opts;
	
	get_opts(argc,argv,&opts);

	
	if(opts.list_metrics){
		printMetrics();
	}
	if(opts.field_codes!=NULL ){
		cout<<opts.field_codes<< endl;
	}
	
	
	
	cout<<opts.list_metrics<< " " <<  opts.number_kernel_launches<<endl;
	cout<<"done"<<endl;
	return 0;
}

void get_opts(int argc, char **argv, struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--output_file or -o <file_path>." << std::endl;
		std::cout << "\t--metric_field_codes or -c. Provide comma separated field codes." << std::endl;
		std::cout << "\t--list_metrics or -l. Provides list of field code to metric mapping." << std::endl;
		std::cout << "\t--number_kernel_launches or -k. Number of kernel launches before ending the session." << std::endl;
        exit(0);
    }


    struct option l_opts[] = {
        {"output_file", no_argument, NULL, 'o'},
        {"metric_field_codes", no_argument, NULL, 'c'},
        {"list_metrics", no_argument, NULL, 'l'},
        {"number_kernel_launches", no_argument, NULL, 'k'}
	
    };

    int ind, c;
	opts->list_metrics =false;
	opts->number_kernel_launches=1;
	opts->out_file=NULL;
	opts->field_codes=NULL;
	
    while ((c = getopt_long(argc, argv, ":o:c:k:l", l_opts, &ind)) != -1)
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
			case ':':
				std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
				exit(1); 
        }
    }
	/*for(; ind < argc; ind++){ //when some extra arguments are passed
      printf("Given extra arguments: %s\n", argv[ind]);
    }*/
}