#pragma once

#include "cupti_callbacks.h"
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include "../include/profileSession.h"

class cupProfiler{
public:
	cupProfiler();
	~cupProfiler();
	
	void cupSubscribe();
	void startSession();
	void endSession();
	
private:
	static void cupCB(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata);
	void initializeContextData(CUcontext &ctx);
	CUpti_SubscriberHandle cupSubscriber;
	std::mutex ctx_mutex;
	std::unordered_map<CUcontext, ctxProfilerData>  ctxProfilerData_map;

	
};

//static void scupCB();