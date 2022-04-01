#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "dlfcn.h" // dlsym, RTLD_NEXT
#include "../include/cuptiErrorCheck.h"
#include "../include/cuptiMetrics.h"
#include "../include/profileSession.h"

#include <vector>
#include <mutex>
#include <string>

#include "cupti_driver_cbid.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"




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

static bool injectionInitialized = false;

ProfileSession profSession;
extern "C" DLLEXPORT int InitializeInjection()
{

	
	if (injectionInitialized == false)
	{
		injectionInitialized = true;

		Metrics::metricList mlist = Metrics::get_metricList();

		for( Metrics::metricList::iterator i= mlist.begin(); i != mlist.end(); i++){
			//profData->metricNames.push_back(i->second);
			profSession.addMetric(i->second);
		}
		
		profSession.subscribeCB();

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

