# cupti_profile

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
The shared libray must be preloaded when launching the executable you would like to profile i.e.  
      
      env LD_PRELOAD=./build/libcuProfile.so LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd` <myexe>
      
I have provided a run.sh file that can be used as an example or as a launching script:

      ./run.sh <myexe>
Some modification to file paths may be required


