# Cupti_profiler

## Dependenccies/ Requirements 
- Cuda Toolkit
- Unix (Tested on Ubuntu)

## Build
You may need to modify the make file to point to the right CUDA and CUPTI directories

- git clone https://github.com/rafsalas19/cupti_profile.git
- cd cupti_profile
- make

This will generate a shared library /build/libcuProfile.so 
It will also generate an executable /build/cupti_profile (this is still in the development phase but can list avalable metrics).

## Use

###### Loading + Running application#
The shared libray must be preloaded when launching the executable you would like to profile i.e.  
      
      env LD_PRELOAD=./build/libcuProfile.so LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd` <myexe>
      
I have provided a run.sh file that can be used as an example or as a launching script:

      ./run.sh <myexe>
Some modification to file paths may be required

###### Controlling metrics and kernel launches
Set these environmental variables to control profiling
- "CUPTI_METRIC_CODES" - Provide comma separated metric codes for the applicable metrics you would like to see.
- "INJECTION_KERNEL_COUNT" - Provide the number of kernel launches you would like to profile per CUDA context.



## Kernel codes and Supported metrics:
| Code         | Metric    | Code       | Metric     |  Code      | Metric     |
|--------------|-----------|------------|------------|------------|------------|
|1000|"achieved_occupancy"|1001|"branch_efficiency"|1002|"cf_executed"|
|1003|"double_precision_fu_utilization"|1004|"dram_read_bytes"|1005|"dram_read_throughput"|
|1006|"dram_read_transactions"|1007|"dram_utilization"|1008|"dram_write_bytes"|1009|"dram_write_throughput"|
|1010|"dram_write_transactions"|1011|"eligible_warps_per_cycle"|1012|"flop_count_dp_add"|
|1013|"flop_count_dp_fma"|1014|"flop_count_dp_mul"|1015|"flop_count_hp_add"|
|1016|"flop_count_hp_fma"|1017|"flop_count_hp_mul"|1018|"flop_count_sp_add"|
|1019|"flop_count_sp_fma"|1020|"flop_count_sp_mul"|1021|"flop_dp_efficiency"|
|1022|"flop_hp_efficiency"|1023|"flop_sp_efficiency"|1024|"gld_efficiency"|
|1025|"gld_throughput"|1026|"gld_transactions"|1027|"gld_transactions_per_request"|
|1028|"global_atomic_requests"|1029|"global_load_requests"|1030|"global_reduction_requests"|
|1031|"global_store_requests"|1032|"gst_efficiency"|1033|"gst_throughput"|
|1034|"gst_transactions"|1035|"gst_transactions_per_request"|1036|"half_precision_fu_utilization"|
|1037|"inst_bit_convert"|1038|"inst_compute_ld_st"|1039|"inst_control"|
|1040|"inst_executed"|1041|"inst_executed_global_atomics"|1042|"inst_executed_global_loads"|
|1043|"inst_executed_global_reductions"|1044|"inst_executed_global_stores"|1045|"inst_executed_local_loads"|
|1046|"inst_executed_local_stores"|1047|"inst_executed_shared_loads"|1048|"inst_executed_shared_stores"|
|1049|"inst_executed_surface_atomics"|1050|"inst_executed_surface_reductions"|1051|"inst_executed_surface_stores"|
|1052|"inst_executed_tex_ops"|1053|"inst_fp_16"|1054|"inst_fp_32"|
|1055|"inst_fp_64"|1056|"inst_integer"|1057|"inst_inter_thread_communication"|
|1058|"inst_issued"|1059|"inst_misc"|1060|"inst_per_warp"|
|1061|"ipc"|1062|"issue_slot_utilization"|1063|"issue_slots"|
|1064|"issued_ipc"|1065|"l1_sm_lg_utilization"|1066|"l2_atomic_throughput"|
|1067|"l2_atomic_transactions"|1068|"l2_global_atomic_store_bytes"|1069|"l2_global_load_bytes"|
|1070|"l2_local_load_bytes"|1071|"l2_read_throughput"|1072|"l2_read_transactions"|
|1073|"l2_surface_load_bytes"|1074|"l2_surface_store_bytes"|1075|"l2_tex_hit_rate"|
|1076|"l2_tex_read_hit_rate"|1077|"l2_tex_read_throughput"|1078|"l2_tex_read_transactions"|
|1079|"l2_tex_write_hit_rate"|1080|"l2_tex_write_throughput"|1081|"l2_tex_write_transactions"|
|1082|"l2_utilization"|1083|"l2_write_throughput"|1084|"l2_write_transactions"|
|1085|"ldst_fu_utilization"|1086|"local_load_requests"|1087|"local_load_throughput"|
|1088|"local_load_transactions"|1089|"local_load_transactions_per_request"|1090|"local_store_requests"|
|1091|"local_store_throughput"|1092|"local_store_transactions"|1093|"local_store_transactions_per_request"|
|1094|"pcie_total_data_received"|1095|"pcie_total_data_transmitted"|1096|"shared_efficiency"|
|1097|"shared_load_throughput"|1098|"shared_load_transactions"|1099|"shared_store_throughput"|
|1100|"shared_store_transactions"|1101|"shared_utilization"|1102|"single_precision_fu_utilization"|
|1103|"sm_efficiency"|1104|"sm_tex_utilization"|1105|"special_fu_utilization"|
|1106|"stall_constant_memory_dependency"|1107|"stall_inst_fetch"|1108|"stall_memory_dependency"|
|1109|"stall_not_selected"|1110|"stall_sleeping"|1111|"stall_texture"|
|1112|"surface_atomic_requests"|1113|"surface_load_requests"|1114|"surface_reduction_requests"|
|1115|"surface_store_requests"|1116|"sysmem_read_bytes"|1117|"sysmem_read_throughput"|
|1118|"sysmem_read_transactions"|1119|"sysmem_write_bytes"|1120|"sysmem_write_throughput"|
|1121|"sysmem_write_transactions"|1122|"tensor_precision_fu_utilization"|1123|"tex_cache_hit_rate"|
|1124|"tex_fu_utilization"|1125|"tex_sm_tex_utilization"|1126|"tex_sm_utilization"|
|1127|"texture_load_requests"|1128|"warp_execution_efficiency"|1129|"warp_nonpred_execution_efficiency"	

