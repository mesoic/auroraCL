// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_interface.cpp
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

// OpenCL Definitions
#define CL_TARGET_ARCH CL_DEVICE_TYPE_GPU
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

// Matrix CL definitions
#define KERNEL_DEFAULT_THREAD_BLOCK_SIZE 16
#define KERNEL_DEFAULT_THREAD_BLOCK_NDR1 cl::NDRange(16)
#define KERNEL_DEFAULT_THREAD_BLOCK_NDR2 cl::NDRange(16, 16)
#define KERNEL_DEFAULT_THREAD_BLOCK_NDR3 cl::NDRange(16, 16, 16)

// Include OpenCL.
#include <CL/cl2.hpp>

// Enable double precision
#if defined(cl_khr_fp64)  	// Khronos extension available?
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #warning "Double precision floating point not supported by OpenCL implementation."
#endif

// Include cl_device wrapper class
#include "cl_device.cpp"

// Container class for representing openCL interfaces  
class cl_interface { 

	public: 

		// OpenCL list platforms
		std::vector<cl::Platform> cl_platforms;

		// Constructors
		cl_interface(void);
		~cl_interface(void);

		// Error string handler
		const char* get_error_string(cl_int);

		// Get methods
		std::vector<cl::Device> get_devices(cl::Platform);
		cl_device get_device(size_t platform_id = 0, size_t device_id = 0);
	
		// Methods
		void show_resources(void);
		void show_platforms(void);
		void show_devices(cl::Platform);

		// Show info for specific platform/device
		void show_platform(cl::Platform);
		void show_platform(size_t);
		void show_device(size_t, size_t);
};

// Destructor
cl_interface::~cl_interface(void) { }

// Constructor
cl_interface::cl_interface(void){
	cl::Platform::get(&this->cl_platforms);
	if ( this->cl_platforms.size() == 0 ){
		std::cout<<" No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
}

// Error strings defined in cl_error.cpp
const char* cl_interface::get_error_string(cl_int error){
	#include "./cl_error.cpp"
}

// Return vector of device objects for a given platform
std::vector<cl::Device> cl_interface::get_devices(cl::Platform platform){

	std::vector<cl::Device> cl_devices;
	platform.getDevices(CL_TARGET_ARCH, &cl_devices);

	if(cl_devices.size()==0){
		std::cout<<" No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	else{
		return cl_devices;
	}
}

// Return cl_device object for a given platfrom
cl_device cl_interface::get_device(size_t platform_id, size_t device_id){

	// Check valid platform_id
	if (  platform_id >= this->cl_platforms.size() ){
		printf("Interface Error: Platform index (%d) not available\n", (int)platform_id);
		exit(1);
	}

	// Available  devices on platform
	std::vector<cl::Device> cl_devices;
	this->cl_platforms[platform_id].getDevices(CL_TARGET_ARCH, &cl_devices);
		
	// Check valid device id
	if ( device_id >= cl_devices.size() ){
		printf("Interface Error: Device index (%d) not available on platform (%d)\n", (int)device_id, (int)platform_id);
		exit(1);
	}

	cl_device device( cl_devices[device_id] );
	return device;
}


// Show all device platforms and resources
void cl_interface::show_resources(void){

	for ( cl::Platform p : this->cl_platforms ) {
		this->show_platform(p);
		this->show_devices(p);
	}
}

// Show available platforms
void cl_interface::show_platforms(void){

	for ( cl::Platform p : this->cl_platforms ) {
		this->show_platform(p);
	}
}

// Show available devices for a given platform
void cl_interface::show_devices(cl::Platform p){

	for (cl::Device d : this->get_devices(p) ){
		cl_device(d).show_device(d);
	}
}

// Wrapper to show OpenCL device by int
void cl_interface::show_device(size_t platform_id, size_t device_id){
	
	if ( platform_id >= ( this->cl_platforms.size() ) ){ 
		printf("Platform (%d) not found\n", (int)platform_id);
	}
	else {
		cl::Platform p = this->cl_platforms[ platform_id ];
		std::vector<cl::Device> devices = this->get_devices(p);

		if ( device_id >= devices.size() ){
			printf("Device (%d) not found on Platform (%d)\n", (int)device_id, (int)platform_id);		
		}
		else {
			cl::Device d = devices[device_id];
			cl_device(d).show_device(d);
		}
	}
}

// Wrapper to show OpenCL platform by int
void cl_interface::show_platform(size_t platform_id){
	
	if ( platform_id >= this->cl_platforms.size() ){
		printf("Platform (%d) not found\n", (int)platform_id);
	}
	else {
		cl::Platform p = this->cl_platforms[ platform_id ];
		this->show_platform(p);
	}
}

// Method to show available OpenCL platforms
void cl_interface::show_platform(cl::Platform platform){

	// Platform Name
	std::cout<< "Platform | "<<platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

	// Platform Vendor
	std::string s;
	platform.getInfo(CL_PLATFORM_VENDOR, &s);
	std::cout << "\t | Vendor\t\t: " << s << "\n";

	// Platform Devices
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	std::cout << "\t | Devices\t\t: " << devices.size() << "\n" << "\n";
}