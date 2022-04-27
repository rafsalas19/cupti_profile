#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "dlfcn.h" // dlsym, RTLD_NEXT
#include "../include/cuptiErrorCheck.h"
#include "../include/cuptiMetrics.h"
#include "../include/profileSession.h"
#include "../include/PWMetrics.h"

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


void deserialize_codes(string &str,vector<int> &tokens){
	while(str.size()> 1){
		
		int end =str.find(",");
		if(end<3){
			end=str.size();
		}
		string tmp =str.substr(0, end);
		int code;
		try{
			cout<<tmp<<endl;
			code=stoi(tmp);	
			if(CODE_MAP_MIN_KEY > code || CODE_MAP_MAX_KEY < code){
				throw std::runtime_error("Code is out of bounds.");
			}
			tokens.push_back(code);
		}
        catch (const std::invalid_argument & e) {
            std::cout << e.what() << ". Invalid code format "+tmp +". Skipping. \n";
			
        }
        catch (const std::out_of_range & e) {
            std::cout << e.what() << ". Invalid code format "+tmp +" Skipping. \n";
        }
		catch(exception& e ){
			 std::cout << e.what() << " Skipping. \n";
		}
		
		if(end==str.size()) break;
		str=str.substr(end+1,str.size());

	}
}




extern "C" DLLEXPORT int InitializeInjection()
{

	
	if (injectionInitialized == false)
	{
		injectionInitialized = true;
		
		char * env_var = getenv("CUPTI_METRIC_CODES");
		if(env_var!=NULL){
			string str =env_var;
			vector<int> metricCodes;
			deserialize_codes(str,metricCodes);
			
			for (auto& code : metricCodes){
				try{
					cupMetrics.getMetricVector()->push_back(cupMetrics.getFormula(code));
				}
				catch(exception& e ){
					std::cout << e.what() << " Skipping. \n";
					continue;
				}			
			}
		}
		if(env_var==NULL ||cupMetrics.getMetricVector()->size()==0){
			cout<< "Metrics not provided to env variable CUPTI_METRIC_CODES, running default metric collection"<<endl;
			Metrics::metricList mlist = Metrics::get_metricList();
	
			for( Metrics::metricList::iterator i= mlist.begin(); i != mlist.end(); i++){
				cupMetrics.getMetricVector()->push_back(i->second);
			}			
		}
		ProfileSession::subscribeCB();

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

