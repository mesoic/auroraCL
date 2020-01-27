// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_device.cpp
//	Copyright (C) 2020 Michael Winters
//	github: https://github.com/mesoic
//	email:  mesoic@protonmail.com
//---------------------------------------------------------------------------------
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//	
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//	
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.
//

// Standard libraries
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <algorithm>

// Include OpenCL.
#include <CL/cl2.hpp>

// Include kernel pre-processor
#include "../pkp/cl_pkp.cpp"

class cl_device {

	public:
		// Device data member
		cl::Device device;
		cl::Context context;

		// Compute kernel
		cl_pkp kernels;
		cl::Program program;  

		// Constructors
		cl_device(cl::Device);
		cl_device(void);
		~cl_device(void);

		// Error string handler
		const char* get_error_string(cl_int);

		// Build sources method
		void kernel_source(const char*);
		void build_sources(void);

		// Get (compiled) kernel object 
		cl::Kernel get_kernel(const char*);

		// Show device methods
		void show_device();
		void show_device(cl::Device);
};

// Null Constructor
cl_device::cl_device(void) { }

// Destructor
cl_device::~cl_device(void) { }

// Constructor
cl_device::cl_device( cl::Device device )
{
	// Device handle
	this->device = device;

	// Establish context (runtime link)
	cl::Context context({this->device});
 	this->context = context;
}

// Error strings defined in cl_error.cpp
const char* cl_device::get_error_string(cl_int error){
	#include "./cl_error.cpp"
}

// Kernel source
void cl_device::kernel_source(const char* file){ 
	this->kernels = cl_pkp(file); 
	this->kernels.pkp_compile_all();
}

// Method to build kernel sources
void cl_device::build_sources(void){

	// Load digest as kernel source
	cl::Program::Sources sources;
	std::string SOURCE = this->kernels.get_digest();
	sources.push_back({SOURCE.c_str(), SOURCE.length()});

	// cl::Program declared here for to maintain catch block scope
	cl::Program program(this->context, sources);

	// Try to build kernel and assign data member
	try {
		program.build({this->device});
		this->program = program;
	}

	// If build fails then report compile errors 	
	catch (cl::Error& e) {

		printf("Build Error(%d): %s\n", e.err(), this->get_error_string( e.err() ) );
		printf("  what(): %s\n", e.what() );
		
		std::cerr<<"\n"<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
		exit(1);
	}	
}

// Method to return compuled kernel source for enqueueNDR
cl::Kernel cl_device::get_kernel(const char* kernel_name){
 	return cl::Kernel(this->program, kernel_name);
}

// Wrapper method for below
void cl_device::show_device( void ){ this->show_device( this->device ); }

// Method to show device on a given platform
void cl_device::show_device( cl::Device device ){

	// Device configuration data 
	std::cout<< "Device\t | "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";
	std::cout<< "\t | C version\t\t: "<<device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()<<"\n";

	// Device __global memory size
	size_t size;
	device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
	std::cout << "\t | __global Mem Size\t: " << size/(1024*1024) << " MB" << "\n";

	// Device __global memory allocation
	device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
	std::cout << "\t | __global Max Alloc\t: " << size/(1024*1024) << " MB" << "\n";

	// Device __local memory size (thread block)
	device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	std::cout << "\t | __local Mem Size\t: " << size/1024 << " KB" << "\n";

	// Device workgroup size
	device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
	std::cout << "\t | Max Workgroup Size\t: " << size << "\n";

	// Workgroup dimensions
	std::vector<size_t> d;
	device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
    std::cout << "\t | Max Workgroup Dims\t: (";
	for ( size_t st : d ) std::cout << ' ' << st;
	std::cout << " )" << "\n";

	// Device available extensions 
	std::vector<std::string> ext_v; 
	std::string ext_all;

	device.getInfo(CL_DEVICE_EXTENSIONS, &ext_all);
	char* ext_k = strtok( &ext_all[0], " ");

	while ( ext_k ){ 
		std::string ext_s( (const char*)ext_k );
		ext_v.push_back(ext_s);
		ext_k = strtok(NULL, " ");
	}

	std::cout << "\t | Device Extensions\t:\n";
	for ( std::string s : ext_v ) std::cout<<"\t\t:= "<<s<<"\n";
	

	// Check for double support
	if ( std::find( ext_v.begin(), ext_v.end(), "cl_khr_fp64") == ext_v.end() ) {
		std::cout<<"\t | Double precision NOT supported\n\n";
	}
	else {
		std::cout<<"\n";
	}
}
