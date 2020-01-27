// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_pkp.cpp
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

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include <map>

// Include CL kerenel
#include "./cl_src.cpp"

// Kernel preprocessor
class cl_pkp {

	public:

		// Kernel path and kernel source objects
		const char* kernel_path;
		std::map<std::string, cl_src> kernels;

		// Kernel names and kernel pkp string
		std::vector<std::string> kernel_names;
		std::string kernel_digest;

		// Constructor
		cl_pkp(const char*);
		cl_pkp(void);
		~cl_pkp(void);

		// Show kerenel wrappers
		void show_source(std::string);
		void show_kernel(std::string);
		void show_config(std::string);

		// Methods to update kernel pkp values
		void update_config(std::string, std::string, std::string);

		// Method to pre-process all kernels and show digest
		void pkp_compile_all(void);
		void show_digest(void);
		
		// Retrieve the digest
		std::string get_digest(void);

		// Method to pre-process single kernel
		void pkp_compile(std::string); 

		// Retrieve kernel source object
		cl_src get_source_object(std::string);
};

cl_pkp::~cl_pkp(void) { }

cl_pkp::cl_pkp(void) { }

// The PKP reads .cl files with one or more defined kernels and translates 
// them into a map of indexable kernel objects. The cl_src and cl_pkp 
// interface together enable <dynamic> compile time constants.
cl_pkp::cl_pkp(const char* path){

	// File pointer
	std::fstream f;

	// Kernel source path
	this->kernel_path = path;
	f.open(path, std::fstream::in);
	std::string line;

	// Parse the source file
	bool is_header = true;
	std::map<std::string, std::string> config_pkp; 
	std::string kernel_buf;
	std::string kernel_name;

	if ( f.is_open() ){

		while( std::getline(f, line) ){

			if ( line.size() != 0 ){

				// Create some buffers
				std::string token_buf = line;
				char* __token = strtok( &token_buf[0], " \t"); 

				// Check if begin kernel
				if ( __token && ( strcmp(__token, "__kernel") == 0 ) ){

					// Write previous kernel to data structure
					if ( !kernel_name.empty() ) {
						
						cl_src kernel(kernel_buf, config_pkp);
						this->kernels[ kernel_name ] = kernel;
						this->kernel_names.push_back( kernel_name );
					}

					// Parse function declaration				
					std::regex r("\\s*(\\w+)\\s*\\(");
					std::smatch m;
				    std::regex_search(line, m, r);
				    kernel_name = m[1];

				    // Zero kernel buffer
					is_header = false;
				    kernel_buf = "\0";
				}

				// Append line to kernel buffer
				if ( !(is_header) &&
					 !(line.empty()) &&
					 !(line.find_first_not_of('\t') == std::string::npos ) &&
					 !(std::regex_search( line , std::regex("\\s*\\/\\/") ) ) ){ 

					std::smatch m;
					std::regex r("#pragma\\s+PKP\\s+(\\w+)\\s*(__default\\s+(\\w))?");

				    if ( std::regex_search(line, m, r ) ){
				    	if ( m.size() == 1 ){ config_pkp[ m[1] ] = "__undefined"; }
				    	if ( m.size() == 4 ){ 
				    		config_pkp[ m[1] ] = m[3]; 
				    	}
					};
				    
					kernel_buf.append( line );
					kernel_buf.append( "\n" );
				}
			}
		}
		// Append last kernel
		cl_src kernel( kernel_buf, config_pkp );
		this->kernels[ kernel_name ] = kernel;
		this->kernel_names.push_back( kernel_name );
	}
	else {
		printf("PKP Error:\n\t(filename) Kernel file (%s) not found\n", this->kernel_path);
		exit(1);
	}
}

// Method to return the kernel digest
std::string cl_pkp::get_digest(void){ return this->kernel_digest; }

// Method to show the kernel digest
void cl_pkp::show_digest(void){	std::cout<<this->kernel_digest<<"\n"; }

// Wrapper for cl_src.show_source()
void cl_pkp::show_source(std::string kernel_name){

	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			this->kernels[ kernel_name ].show_source();
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error: Key (%s) not Found \n", kernel_name.c_str() );
		exit(1);
	}
}

// Wrapper for cl_src.show_source()
void cl_pkp::show_kernel(std::string kernel_name){

	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			this->kernels[ kernel_name ].show_kernel();
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error: Key (%s) not found \n", kernel_name.c_str() );
		exit(1);
	}
}

// Wrapper for cl_src.show_config()
void cl_pkp::show_config(std::string kernel_name){

	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			this->kernels[ kernel_name ].show_config();
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error: Key (%s) not found \n", kernel_name.c_str() );
		exit(1);
	}
}

// Wrapper for cl_src.update_config()
void cl_pkp::update_config(std::string kernel_name, std::string __constant, std::string __value){

	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			this->kernels[ kernel_name ].update_config(__constant, __value);
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error:'\n\t(update) Key (%s) not found \n", kernel_name.c_str() );
		exit(1);
	}
}

// Build all kernels
void cl_pkp::pkp_compile_all(void){

	this->kernel_digest.clear();
	for ( std::string kernel_name : this->kernel_names ){
		this->pkp_compile( kernel_name );
		this->kernel_digest.append( this->kernels[ kernel_name ].kernel_pkp );
	}
}

// Build single kernel
void cl_pkp::pkp_compile(std::string kernel_name){
	
	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			this->kernels[ kernel_name ].pkp_compile();
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error:\n\t(compile) Key (%s) not found \n", kernel_name.c_str() );
		exit(1);
	}
}

// Method to retrieve single kernel from pkp
cl_src cl_pkp::get_source_object(std::string kernel_name){
	
	try {
		if ( this->kernels.find( kernel_name ) == this->kernels.end() )
			throw std::invalid_argument("");
		else 
			return this->kernels[ kernel_name ];
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Error:\n\t(source object) Key (%s) not found \n", kernel_name.c_str() );
		exit(1);
	}
}