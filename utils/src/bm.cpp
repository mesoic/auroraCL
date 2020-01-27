// ---------------------------------------------------------------------------------
//	auroraCL -> utils/src/cl_mmul.cpp
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

// Define target device
#define KERNEL_FILE_f32 "../../kernels/f32/cl_product_f32.cl"
#define KERNEL_DEFAULT_BLOCK_SIZE 8
#define KERNEL_MAX_BLOCK_SIZE 20

// Define target device
#define PLATFORM_ID 0 
#define DEVICE_ID 0

// Include cl_interface class
#include "../../lib/interface/cl_interface.cpp"
#include "../../lib/utils/cl_time.cpp"
#include "../../lib/utils/cl_parse.cpp"
#include "../../inc/cl_matrix.hpp"

typedef struct{
	int D_MIN;
	int D_MAX; 
	int D_SIZE;
	int B_SIZE; 
	int CYCLES; 
} cl_bm_config;

class cl_bm_cli {

	public:
		
		// Initialize domain and range data structures
		std::vector<size_t> domain; 
		cl_bm_config config; 

		// cl_matrix objects
		cl_matrix<float> A; 
		cl_matrix<float> B; 

		// some structures to store results
		std::map<size_t, std::vector<cl_time::cl_time_t>> map_t;

		// Matrix data fill format
		bool fill_index = false;

		// Constructor/Destructor
		cl_bm_cli(cl_interface, cl_bm_config);
		~cl_bm_cli(void);

		// Hardware acceleration objects
		cl_interface interface;
		cl_device GPU;

		// Target run CPU for comparison
		bool run_cpu = false; 
		bool pprint  = false;

		// Develop block domain
		void logarithmic_block_domain(size_t, size_t, size_t, size_t, size_t);
		void print_domain();

		// Benchmark methods
		void probe_scaling(void);
		void probe_blocksize(void);

		// Write file data
		std::string header; // data header
		void write_file(std::string);

}; 

// Destructor
cl_bm_cli::~cl_bm_cli(void) { }

// Constructor
cl_bm_cli::cl_bm_cli(cl_interface interface, cl_bm_config config){


	// Store configuration data structure
	this->interface = interface;
	this->config = config;

	// Calculate domain 
	this->logarithmic_block_domain(
		config.D_MIN,
		config.D_MAX,
		config.D_SIZE,
		config.B_SIZE, 
		2
	);

	// print domain
	this->print_domain();

	// If we have gotten here then all tests passed. Fire up the kernel
	this->GPU = interface.get_device( PLATFORM_ID, DEVICE_ID );
	this->GPU.kernel_source(KERNEL_FILE_f32);
	this->GPU.kernels.update_config("f32_product_v2", "WORK_PER_THREAD_N", std::to_string( config.B_SIZE ) );
	this->GPU.kernels.pkp_compile_all();
	this->GPU.build_sources();
}

// Block logspace function = BLOCK_SIZE*logspace()
void cl_bm_cli::logarithmic_block_domain(
	size_t start, 
	size_t stop, 
	size_t num = 64,  
	size_t blocksize = KERNEL_MAX_BLOCK_SIZE,
	size_t base = 2 ) 
{

 	float rStart = pow( (float)base, (float)start);
 	float rBase  = pow( (float)base, ((float)stop-(float)start)/(float)num);

 	std::vector<size_t> vals;
 	vals.reserve(num);

 	for ( size_t n = 0; n < num; n++ ){
 		rStart*=rBase;
 		vals.push_back( (size_t)rStart * blocksize );
 	}
	
 	vals.erase( unique( vals.begin(), vals.end() ), vals.end() );
 	this->domain.reserve( vals.size() * sizeof(size_t) );
 	this->domain = vals;
}

// Print domain function 
void cl_bm_cli::print_domain(void){

	int count = 0;	
	printf("\t| Domain(calc)\t\t= [[\n");
	for ( size_t N : this->domain ){
		if (count%16 == 0){ printf("\t|\t"); }
		printf( "%d ", (int)N );
		count++;
		if (count%16 == 0){ printf("\n"); }		
	}
	printf(" ]]\n");
}


// Scaling test
void cl_bm_cli::probe_scaling(void){

	// Loop through all domain values
	for ( size_t N : this->domain ){

		// Vector to store time objects
		cl_time s; 
		std::vector<cl_time::cl_time_t> vec_t; 

		cl_matrix<float> A(N, N);
		cl_matrix<float> B(N, N);

		A.fill_rand(1,10,10);
		B.fill_rand(1,10,10);

		printf("N=%d\t|\n", (int)N);

		// For each kernel 
		for ( std::string k_name : this->GPU.kernels.kernel_names ){

			// Run a certain number of multiply cycles
			for ( size_t i = 0; i < (size_t)this->config.CYCLES; i++){

				s.start();
				cl_matrix<float> C = A.product(B, this->GPU, k_name.c_str(), 
					cl::NDRange(this->config.B_SIZE, this->config.B_SIZE));				
				s.end();				
				vec_t.push_back(s.delta());
				if (this->pprint){
					printf("\t| %s\t %fus\n", k_name.c_str(), s.delta().count() );
				}
			}
		}

		// Run single cycle CPU 
		if (this->run_cpu){
			s.start();
			cl_matrix<float> C = A.product(B);
			s.end();
			vec_t.push_back(s.delta());
			if (this->pprint){
				printf("\t| %s\t\t\t %fus\n", "CPU", s.delta().count() );
			}
		}

		// Push back data for N 
		this->map_t[N] = vec_t;
	}

	// Prepare file header
	this->header.append("N\t"); int count = 0;
	for ( std::string k_name : this->GPU.kernels.kernel_names ){

		for ( size_t i = 0; i < (size_t)this->config.CYCLES; i++){

			std::string col = std::to_string(count);
			col.append(":");
			col.append(std::to_string(i)); 
			col.append("\t\t");
			this->header.append(col);
		}
		count++;
	}

	// If running CPU, add to column to header
	if (this->run_cpu){
		this->header.append("CPU\n");
	}
	else{
		this->header.append("\n");
	}
}


// Probe blocksize scaling against 'naive kernel'
void cl_bm_cli::probe_blocksize(void){

	// NDR optimization
	std::vector<cl::NDRange> NDR; 
	
	// Blocksize tests
	NDR.push_back( cl::NDRange(2,2) );
	NDR.push_back( cl::NDRange(4,4) );
	NDR.push_back( cl::NDRange(8,8) );
	NDR.push_back( cl::NDRange(16,16) );
	NDR.push_back( cl::NDRange(32,8) );
	NDR.push_back( cl::NDRange(64,4) );
	NDR.push_back( cl::NDRange(128,2) );
	NDR.push_back( cl::NDRange(256,1) );

	// Kernel name (should be naive kernel)
	std::string k_name = this->GPU.kernels.kernel_names[0];

	// Loop through all domain values
	for ( size_t N : this->domain ){

		// Vector to store time objects
		cl_time s; 
		std::vector<cl_time::cl_time_t> vec_t; 

		cl_matrix<float> A(N, N);
		cl_matrix<float> B(N, N);

		A.fill_rand(1,10,10);
		B.fill_rand(1,10,10);

		printf("N=%d\t|\n", (int)N);

		// Loop through blocksizes
		for( cl::NDRange ndr : NDR ) {

			s.start();
			cl_matrix<float> C = A.product(B, this->GPU, k_name.c_str(), ndr );
			s.end();
			vec_t.push_back(s.delta());
			if (this->pprint){
				printf("\t| NDR(%d:%d)\t %fus\n", (int)ndr[0], (int)ndr[1], s.delta().count() );
			}

		}

		// Run CPU for comparison if desired
		if (this->run_cpu){
			s.start();
			cl_matrix<float> C = A.product(B);
			s.end();
			vec_t.push_back(s.delta());
			if (this->pprint){
				printf("\t| %s\t\t %fus\n", "CPU", s.delta().count() );
			}
		}

		// Push back data
		this->map_t[N] = vec_t;
	}

	// Prepare header
	this->header.append("N\t"); 
	for( cl::NDRange ndr : NDR ){

		std::string col = std::to_string(ndr[0]);
		col.append(":");
		col.append( std::to_string(ndr[1]));
		col.append("\t\t");
		this->header.append(col);
	}
	
	// If running CPU, add to column to header
	if (this->run_cpu){
		this->header.append("CPU\n");
	}
	else{
		this->header.append("\n");
	}
}

// Write output data
void cl_bm_cli::write_file(std::string filename){

	// Write output
	std::fstream f;
	f.open( filename.c_str(), std::fstream::out );
	if ( f.is_open() ){

		// Output header
		f<<this->header;		
	
		// Write out data from unit test
		for ( size_t N : this->domain ){
			
			f<<N<<"\t"; 
			for( cl_time::cl_time_t time_t : this->map_t[N]){
		 		f<< time_t.count() <<"\t\t";
		 	} 
		 	f<<"\n";
		}
		f.close();
	}
}


// Main program
int main(int argc, char** argv){

	printf("\n\t------------------------------------------------\n");
	printf("\t| AuroraCL Matrix Multiplication Benchmark CLI |\n");
	printf("\t------------------------------------------------\n");

	// cl_input_parser
	cl_input_parser input(argc, argv);

	// Set up some metadata for the parser
	std::vector<std::string> mode_vals 	= {"scaling", "blocksize"}; 
	std::vector<std::string> num_vals 	= {"3"}; 

	input.add_key_rule("-m", (function)sanitize_in_tuple, mode_vals);
	input.add_key_rule("-d", (function)sanitize_int_list, num_vals);
	input.add_key_rule("-c", (function)sanitize_int);
	input.add_key_rule("-b", (function)sanitize_int);
	input.add_key_rule("-f", (function)sanitize_string);
	input.add_key_rule("-p", (function)sanitize_exists);
	input.add_key_rule("-h", (function)sanitize_exists);
	input.add_key_rule("-cpu", (function)sanitize_exists);
	input.map_key_rules();
 

	// Help method
	if ( input.is_key_passed("-h") ){
		printf("\nCommand Reference\n"); 
		printf("\t | -m(str) \t= Benchmark Mode {\"scaling\", \"blocksize\"} \n");
		printf("\t | -d([int]) \t= Block Logarithmic Domain (min) (max) (npoints) \n");
		printf("\t | -c(int) \t= Number of kernel cycles (scaling mode only) \n");
		printf("\t | -b(int) \t= GPU thread-block size (default = 8) \n");
		printf("\t | -p(void) \t= print marix output during runtime (optional) \n");
		printf("\t | -cpu(void) \t= run CPU (optional) \n");
		
		printf("\nUsage Examples\n"); 
		printf("\t | bmcli -m scaling \t\t\t= Basic scaling test\n");
		printf("\t | bmcli -m scaling -c 4\t\t= Basic scaling test with 4 cycles per GPU kernel\n");
		printf("\t | bmcli -m scaling -p -f <filename>\t= Basic scaling test. Print output and save to file\n");
		printf("\t | bmcli -m scaling -d 0 7 32 -b 4\t= Custom Domain [4*(2**0), 4*(2**7)] with 32 points\n");
		printf("\t | bmcli -m blocksize \t\t\t= Basic blocksize test\n");
		printf("\t | bmcli -m blocksize -d 0 6 64 -b 8 \t= Custom Domain [8*(2**0), 8*(2**6)] with 64 points\n\n");
		return 0;
	}

	// Check that a mode has been passed bmcli
	std::string mode = "\0";
	if ( input.is_key_passed("-m") ) {
		mode = input.get_key_values("-m")[0];
	}
	else{ 
		printf("Input Error: Missing required flag (-m). See -h for usage\n");
		exit(1);
	}

	// Extract cycles variable 
	int cycles = 1;
	if ( input.is_key_passed("-c") ){

		std::vector<std::string> c_key_data = input.get_key_values("-c");
		cycles = std::stoi(c_key_data[0]);

		// Print domain information
		printf("\t| Cycles(user) \t\t= (%d) \n", cycles);
	} 
	else{ 
		printf("\t| Cycles(default) \t= (%d) \n", cycles);
	}


	// Extract blocksize variable
	int b_size = 8;
	if ( input.is_key_passed("-b") ){

		std::vector<std::string> b_key_data = input.get_key_values("-b");
		b_size = std::stoi(b_key_data[0]);

		// Print domain information
		printf("\t| Blocksize(user) \t= (%d) \n", b_size);	
	}
	else {
		printf("\t| Blocksize(default) \t= (%d) \n", b_size);
	}

	
	// Extract domain variables
	int d_min = 0, d_max = 7, d_size = 32; 
	if ( input.is_key_passed("-d") ){

		std::vector<std::string> d_key_data = input.get_key_values("-d");
		d_min  = std::stoi( d_key_data[0] );
		d_max  = std::stoi( d_key_data[1] );
		d_size = std::stoi( d_key_data[2] );

		// Print domain information
		printf("\t| Domain(user) \t\t= logspace(%d, %d, npoints = %d) \n", 
			b_size * (int)std::pow((float)2, (float)d_min), 
			b_size * (int)std::pow((float)2, (float)d_max), 
			d_size
		);
	}
	else {
		printf("\t| Domain(default) \t= logspace(%d, %d, npoints = %d) \n",
			b_size * (int)std::pow((float)2, (float)d_min), 
			b_size * (int)std::pow((float)2, (float)d_max), 
			d_size
		);
	}

	// File output for data
	std::string filename;
	if ( input.is_key_passed("-f") ) {

		std::vector<std::string> f_key_data = input.get_key_values("-f");
		filename = f_key_data[0];
	}

	// If scaling mode
	if ( mode.compare("scaling") == 0 ){	

		// Prepare struct
		cl_interface interface;
		cl_bm_config config;

		config.D_MIN  = d_min;
		config.D_MAX  = d_max;
		config.D_SIZE = d_size;
		config.B_SIZE = b_size;
		config.CYCLES = cycles;

		// Call constructor
		cl_bm_cli bm( interface, config );
			
		// Set CPU and print output variables
		bm.run_cpu = input.is_key_passed("-cpu") ? true : false;
		bm.pprint  = input.is_key_passed("-p") ? true : false; 

		// Run dynamic scaling probe
		bm.probe_scaling();

		// If filename variable has been assigned, write output data
		if( !filename.empty() ){
			bm.write_file( filename );
		}
	}

	// If blocksize mode
	if ( mode.compare("blocksize") == 0 ){	

		// Prepare struct
		cl_interface interface;
		cl_bm_config config;

		config.D_MIN  = d_min;
		config.D_MAX  = d_max;
		config.D_SIZE = d_size;
		config.B_SIZE = b_size;
		config.CYCLES = cycles;

		// Call constructor
		cl_bm_cli bm( interface, config );
			
		// Set CPU and print output variables
		bm.run_cpu = input.is_key_passed("-cpu") ? true : false;
		bm.pprint  = input.is_key_passed("-p") ? true : false; 

		// Run dynamic scaling probe
		bm.probe_blocksize();

		// If filename variable has been assigned, write output data
		if( !filename.empty() ){
			bm.write_file( filename );
		}
	}
}
