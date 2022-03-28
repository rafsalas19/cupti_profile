#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "dlfcn.h" // dlsym, RTLD_NEXT
#include "../include/cuptiErrorCheck.h"
#include "../include/cuptiMetrics.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_driver_cbid.h"
#include "cupti_target.h"
#include "nvperf_host.h"

#include <vector>
#include <mutex>
#include <string>





extern "C"
{
    extern typeof(dlsym) __libc_dlsym;
    extern typeof(dlopen) __libc_dlopen_mode;
}

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

using namespace std;

// List of metrics to collect
vector<string> metricNames;

static bool injectionInitialized = false;

extern "C" DLLEXPORT int InitializeInjection()
{

    if (injectionInitialized == false)
    {
        injectionInitialized = true;

        cout << "Hello world from the injection library" << endl;

	Metrics::metricList mlist = Metrics::get_metricList();

	for( Metrics::metricList::iterator i= mlist.begin(); i != mlist.end(); i++){
		metricNames.push_back(i->second);
	}
	for (int i =0;i<metricNames.size();++i){
		cout<<metricNames[i]<<endl;
	}
    }

    return 1;
}

extern "C" DLLEXPORT void * dlsym(void * handle, char const * symbol)
{
    InitializeInjection();

    typedef void * (*dlsym_fn)(void *, char const *);
    static dlsym_fn real_dlsym = NULL;
    if (real_dlsym == NULL)
    {
        // Use libc internal names to avoid recursive call
        real_dlsym = (dlsym_fn)(__libc_dlsym(__libc_dlopen_mode("libdl.so", RTLD_LAZY), "dlsym"));
    }
    if (real_dlsym == NULL)
    {
        cerr << "Error finding real dlsym symbol" << endl;
        return NULL;
    }
    return real_dlsym(handle, symbol);
}

