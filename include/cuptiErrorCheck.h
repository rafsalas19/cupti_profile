#pragma once
#include <exception>
#include <string>
#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        throw std::runtime_error("Function " + std::string(#apiFuncCall)+" "  +  to_string (_status));        \
    }                                                                          \
} while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
         throw std::runtime_error("Function " + std::string(#apiFuncCall)+" "  +  to_string (_status));       \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        throw std::runtime_error("Function " + std::string(#apiFuncCall)+" "  +  to_string (_status));		   \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          				\
do {                                                                           				\
    cudaError_t _status = apiFuncCall;                                         				\
    if (_status != cudaSuccess) {                                              				\
       throw std::runtime_error("Function " + std::string(#apiFuncCall)+" " +  cudaGetErrorString(_status)); 	\
    }                                                                         				\
} while (0)                                                                     			

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        throw std::runtime_error("Error: Memory Allocation Failed \n");  		\
    }                                                                           \
} while (0)

#define NVPW_ERROR_CHECK(apiFuncCall)                                        						\
do {                                                                                				\
    NVPA_Status _status = apiFuncCall;                                                    			\
    if (NVPA_STATUS_SUCCESS != _status) {                                            				\
        throw std::runtime_error("Failed " + std::string(#apiFuncCall) + " " + to_string(_status)); \
    }                                                                               				\
} while (0)
	