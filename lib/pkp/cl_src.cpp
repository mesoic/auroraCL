// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_src.cpp
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

// Kernel source object
class cl_src {
	
	public:
		std::string kernel_src; // Holds kernel soruce pre-compile 
		std::string kernel_pkp; // Holds kernel source post-compile

		// Map to hold compile time constants
		std::map<std::string, std::string> config_pkp;

		// Constructors/Destructor 
		cl_src(std::string, std::map<std::string, std::string>);
		cl_src(void);
		~cl_src(void);

		// Show source/pkp configuration 
		void show_source(void);
		void show_config(void);
		void show_kernel(void);

		// Methods to update kernel pkp values
		void update_config( std::string, std::string );

		// Run the preprocessor
		void pkp_compile(void);
};

// Constructor
cl_src::cl_src( 
	std::string src, 
	std::map<std::string, std::string> pkp)
{
	this->kernel_src = src;
	this->kernel_pkp = "\0";
	this->config_pkp = pkp;
}

// Empty constructor (needed for map)
cl_src::cl_src( void ) { }

// Destructor
cl_src::~cl_src( void ) { }


// Show kernel source
void cl_src::show_source( void ){ 
	std::cout<<this->kernel_src<<"\n"; 
}

// Show kernel (post pkp compilation)
void cl_src::show_kernel( void ){ 
	std::cout<<this->kernel_pkp<<"\n"; 
}

// Show compile time constants
void cl_src::show_config( void ){
	for(auto it = this->config_pkp.cbegin(); it != this->config_pkp.cend(); ++it){
		std::cout << it->first << " " << it->second << "\n";
	}
}

void cl_src::update_config( std::string __constant, std::string __value ){

	try {
		if ( this->config_pkp.find( __constant ) == this->config_pkp.end() )
			throw std::invalid_argument("");
		else 
			this->config_pkp[ __constant ] = __value;
	}
	catch ( const std::invalid_argument &e) {
		printf("PKP Key (%s) not Found \n", __constant.c_str() );
	}
}

// Kernel preprocessor compile method
void cl_src::pkp_compile( void ){

	std::stringstream s( this->kernel_src );
	std::string line;

	// Kernel string
	std::string kernel = "\0";

	// Run the preprocessor
	std::smatch m;
	while ( std::getline( s, line ) ){
		
		if ( std::regex_search( line, m, std::regex("#pragma\\s+PKP\\s+(\\w+)") ) ){
			char pkp_buffer [128];
			std::sprintf(
				pkp_buffer, 
				"\t#define %s %s", 
				m[1].str().c_str(), 
				this->config_pkp[ m[1].str() ].c_str()
			);
			std::string pkp_define( pkp_buffer );
			kernel.append( pkp_define );
			kernel.append( "\n" );
		}
		else {
			kernel.append( line );
			kernel.append( "\n" );
		}
	}
	this->kernel_pkp = kernel;
}