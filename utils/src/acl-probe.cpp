// ---------------------------------------------------------------------------------
//	auroraCL -> utils/src/cl_probe.cpp
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

// Include cl_interface class
#include "../../lib/interface/cl_interface.cpp"
#include "../../lib/utils/cl_parse.cpp"

// Main program 
int main(int argc, char** argv){

	printf("\n\t ------------------------------------------------\n");
	printf("\t |  	 AuroraCL OpenCL Assets Probe 		|\n");
	printf("\t ------------------------------------------------\n");

	// cl_input_parser
	cl_input_parser input(argc, argv);
	input.add_key_rule("-p",  (function)sanitize_int );
	input.add_key_rule("-d",  (function)sanitize_int );
	input.add_key_rule("-h",  (function)sanitize_exists );
	input.map_key_rules();

	// Help menu
	if ( input.is_key_passed("-h") ){
		printf("\nCommand Reference\n"); 
		printf("\t | -p(int) \t= OpenCL platform ID \n");
		printf("\t | -d(int) \t= OpenCL device ID for platform N \n");

		printf("\nUsage Examples\n"); 
		printf("\t | cl_probe \t\t= Probe <all> system assets\n");
		printf("\t | cl_probe -p 0 \t= Probe data for platform (0)\n");
		printf("\t | cl_probe -p 1 -d 0 \t= Probe data for device (0) on platform (1)\n");
		return 0;
	}

	// cl_input_parser    
	cl_interface interface;
		
	if ( input.no_key_passed() ){ 
		interface.show_resources(); 
	}

	// If only platform is desired
	if ( input.is_key_passed("-p") && !input.is_key_passed("-d") ){
		std::vector<std::string> key_data = input.get_key_values("-p");
		interface.show_platform(std::stoi( key_data[0] ));
	}

	// If probing for a device
	if ( input.is_key_passed("-p") && input.is_key_passed("-d") ){
		std::vector<std::string> p_key_data = input.get_key_values("-p");
		std::vector<std::string> d_key_data = input.get_key_values("-d");
		interface.show_device(std::stoi(p_key_data[0]),std::stoi(d_key_data[0]));
	}
}