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

// Container class
class cl_mmul_demo {

	public:
		// Matrix Dimensions
		size_t M, K, N; 

		// cl_matrix objects
		cl_matrix<float> A; 
		cl_matrix<float> B; 

		// some structures to store results
		std::map<std::string, cl_matrix<float>> C_DATA; 

		// Hardware acceleration objects
		cl_interface interface;
		cl_device GPU;

		// Objects to format data
		bool fill_index = false; 
		
		// Blocksize
		size_t B_SIZE; 
		
		// Constructors/Destructor
		cl_mmul_demo(size_t M, size_t K, size_t N, size_t B_SIZE = KERNEL_DEFAULT_BLOCK_SIZE);
		~cl_mmul_demo(void);

		// Run the kernel
		void gpu_product(void);
		void cpu_product(void);

		// Matrix Equivalence test
		void equivalence_test(void);
		void print_results(void);
};

// Initialize matrices
cl_mmul_demo::cl_mmul_demo(size_t M, size_t K, size_t N, size_t B_SIZE ){
	
	// Store sizes
	this->M = M; 
	this->K = K; 
	this->N = N;

	// Call cl_matrix constructors
	this->A = cl_matrix<float>(M, K);
	this->B = cl_matrix<float>(K, N);

	// Populate initial values
	if ( this->fill_index ){
		this->A.fill_ints();
		this->B.fill_ints();
	}
	else { 
		this->A.fill_rand(1,10,10);
		this->B.fill_rand(1,10,10);
	}

	// Check blocksizes against matrix dimensions
	if ( ( M == K ) && ( K == N ) ) {
		if ( M % B_SIZE != 0 ){
			printf("Warning: Unaligned Blocksize (%d) on dimension (%d) \n", (int)B_SIZE, (int)M );
		} 
	}
	else {
		if ( M % B_SIZE != 0 ){
			printf("Warning: Unaligned Blocksize (%d) on dimension M = (%d) \n", (int)B_SIZE, (int)M );
		}
		if ( K % B_SIZE != 0 ){
			printf("Warning: Unaligned Blocksize (%d) on dimension K = (%d) \n", (int)B_SIZE, (int)K );
		}
		if ( N % B_SIZE != 0 ){
			printf("Warning: Unaligned Blocksize (%d) on dimension N = (%d) \n", (int)B_SIZE, (int)N );
		}
	}

	// Check blcoksize against device maximum blocksize
	if ( B_SIZE > KERNEL_MAX_BLOCK_SIZE ) {
		printf("Error: Blocksize (%d) exceeds maximum blocksize (%d) \n", (int)B_SIZE, (int)KERNEL_MAX_BLOCK_SIZE );
	}

	// If we have gotten here then all tests passed. Fire up the kernel
	this->GPU = this->interface.get_device( PLATFORM_ID, DEVICE_ID );
	this->GPU.kernel_source(KERNEL_FILE_f32);
	this->GPU.kernels.update_config("f32_product_v2", "WORK_PER_THREAD_N", std::to_string( B_SIZE ) );
	this->GPU.kernels.pkp_compile_all();
	this->GPU.build_sources();

	// Assign class blocksize
	this->B_SIZE = B_SIZE;
}

// Destructor
cl_mmul_demo::~cl_mmul_demo(void) { }

// A separate method to run the product routines (GPU)
void cl_mmul_demo::gpu_product(void){

	// Local time object
	cl_time s; 
	
	// Run kernels 
	for ( std::string k_name : this->GPU.kernels.kernel_names ){
		
		// Build result matrix
		cl_matrix<float> C(this->M, this->N);
		
		// Run kernel 
		s.start();
		C = A.product(B, this->GPU, k_name.c_str(), cl::NDRange(this->B_SIZE, this->B_SIZE) );
		s.end();
		printf("Kernel (%s)\n\t Elapsed time: (%fus)\n\n", k_name.c_str(), s.delta().count() );

		// Store data in result matrix
		C_DATA[ k_name ] = C;
	}
}

// A separate method to run the product routines (CPU)
void cl_mmul_demo::cpu_product(void){

	// Local time object
	cl_time s; 
	cl_matrix<float> C(this->M, this->N);

	// Calculate CPU product
	s.start();
	C = A.product(B);
	s.end();
	printf("Kernel (CPU)\n\t Elapsed time: (%fus)\n\n", s.delta().count() );

	// Store data in result matrix
	this->C_DATA[ "CPU" ] = C;
}

// A method to verify equivalence
void cl_mmul_demo::equivalence_test(void) {

	// If the CPU data exists in C_DATA
	cl_matrix<float> C0 = ( this->C_DATA.find("CPU") != this->C_DATA.end() ) ? 
		this->C_DATA[ "CPU" ] : this->C_DATA[ this->GPU.kernels.kernel_names[0] ];

	// Looking for != 
	bool kernel_neq = false;

	// Loop through kernel keys
	for ( std::string k_name : this->GPU.kernels.kernel_names ){

		// If we find inequivalence we must do further tests
		if ( this->C_DATA[k_name] != C0 ){ kernel_neq = true; }
	}

	// Investigate further
	if ( kernel_neq ){

		// Check other matrices
		cl_matrix<float> C1 =  this->C_DATA[ this->GPU.kernels.kernel_names[1] ];
		cl_matrix<float> C2 =  this->C_DATA[ this->GPU.kernels.kernel_names[2] ];

		printf( "Matrix Inequivalence on Kernel \n");
	
		// Boolean permutator test on multiplication kernels 
		if ( ( C0 == C1 ) && ( C1 != C2 ) ){ printf("v0 == v1 != v2\n"); }
		if ( ( C0 == C2 ) && ( C1 != C2 ) ){ printf("v0 == v2 != v1 \n"); }
		if ( ( C1 == C2 ) && ( C1 != C0 ) ){ printf("v0 != v1 == v2 \n"); }
		if ( ( C1 != C2 ) && ( C1 != C0 ) && (C2 != C0) ){ printf("v0 != v1 != v2\n"); }
	}
}

// Method to print results
void cl_mmul_demo::print_results(void) {

	this->A.pprint("A = ");
	this->B.pprint("B = ");

	// Print GPU versions	
	for ( std::string k_name : this->GPU.kernels.kernel_names ){
		printf("\nKernel (%s)\n", k_name.c_str() );
		this->C_DATA[k_name].pprint("C = ");
	}

	// Print CPU version if calculated
	if ( this->C_DATA.find("CPU") != this->C_DATA.end() ) {
		printf("\nKernel (%s)\n", "CPU" );		
		this->C_DATA["CPU"].pprint("C = ");
	}
}

// Main program 
int main(int argc, char** argv){

	printf("\n----------------------------------------------------\n");
	printf("| AuroraCL Accelerated Matrix Multiplication Probe |\n");
	printf("----------------------------------------------------\n\n");
	
	// cl_input_parser
	cl_input_parser input(argc, argv);
	input.add_key_rule("-k", (function)sanitize_int ); // Matrix A(m,k)
	input.add_key_rule("-m", (function)sanitize_int ); // Matrix B(k,n)
	input.add_key_rule("-n", (function)sanitize_int ); // Matric C(m,n)
	input.add_key_rule("-b", (function)sanitize_int ); // Block size
	input.add_key_rule("-h", (function)sanitize_exists);
	input.add_key_rule("-p", (function)sanitize_exists ); // Print result
	input.add_key_rule("-cpu", (function)sanitize_exists ); // Run CPU
	input.map_key_rules();

	// cl_interface  
	cl_interface interface;

	// Help method
	if ( input.is_key_passed("-h") ){
		printf("\nCommand Reference\n"); 
		printf("\t | -n(int) \t= matrix dimension (n) \n");
		printf("\t | -k(int) \t= matrix dimension (k) \n");
		printf("\t | -m(int) \t= matrix dimension (m) \n");
		printf("\t | -b(int) \t= GPU thread-block size (optional) \n");
		printf("\t | -p(void) \t= print marix output (optional) \n");
		printf("\t | -cpu(void) \t= run CPU (optional) \n");

		printf("\nUsage Examples\n"); 
		printf("\t | mmul -n 1024 \t\t= multiply square matrices with A(1024,1024) * B(1024,1024)\n");
		printf("\t | mmul -n 1024 -b 16 \t\t= multiply square matrices with accelerator thread-block size (16)\n");
		printf("\t | mmul -n 1024 -cpu \t\t= multiply square matrices and include CPU benchmark\n");
		printf("\t | mmul -n 8 -p \t\t= multiply A(8,8) * B(8,8) and print result\n");
		printf("\t | mmul -m 32 -k 16 -n 24 \t= multiply non-square matrices with A(32,16) * B(16,24)\n");
		printf("\t | mmul -m 32 -k 16 -n 24 -b 8\t= multiply non-square matrices with custom accelerator thread-blocksize (8)\n\n");
		return 0;
	}

	// Shorthand to multiply square matrices (default blocksize)
	else if ( input.is_key_passed("-n") && 
			 !input.is_key_passed("-k") &&
			 !input.is_key_passed("-m") && 
			 !input.is_key_passed("-b") ) {

		// Parse input values
		std::vector<std::string> n_key_data = input.get_key_values("-n");

		// Check that size is greater than defualt block size
		if ( std::stoi(n_key_data[0]) < KERNEL_DEFAULT_BLOCK_SIZE ){

			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(n_key_data[0]), KERNEL_DEFAULT_BLOCK_SIZE );
			return -1;
		}

		else {

			// Perform GPU multiplication
			cl_mmul_demo mmul( 
				std::stoi(n_key_data[0]), 
				std::stoi(n_key_data[0]), 
				std::stoi(n_key_data[0]) );
			mmul.gpu_product();

			// If CPU flag is passed run product
			if ( input.is_key_passed("-cpu") ){
				mmul.cpu_product();
			}

			// Run equivalence test
			mmul.equivalence_test();

			// Print results if desired
			if ( input.is_key_passed("-p") ) {
				mmul.print_results();
			}
			return 0;
		}
	}

	// Shorthand to multiply square matrices (custom blocksize)
	else if ( input.is_key_passed("-n") && 
			  input.is_key_passed("-b") &&
		 	 !input.is_key_passed("-k") &&
			 !input.is_key_passed("-m") ) {


		// Parse input values
		std::vector<std::string> n_key_data = input.get_key_values("-n");
		std::vector<std::string> b_key_data = input.get_key_values("-b");

		// Check that size is greater than defined block size
		if ( std::stoi(n_key_data[0]) < std::stoi(b_key_data[0]) ){
	
			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(n_key_data[0]), std::stoi(b_key_data[0]));
			return -1;
		}

		else{

			// Perform GPU multiplication
			cl_mmul_demo mmul( 
				std::stoi(n_key_data[0]), 
				std::stoi(n_key_data[0]), 
				std::stoi(n_key_data[0]), 
				std::stoi(b_key_data[0]) );
			mmul.gpu_product();

			// If CPU flag is passed run product
			if ( input.is_key_passed("-cpu") ){
				mmul.cpu_product();
			}

			// Run equivalence test
			mmul.equivalence_test();

			// Print results if desired
			if ( input.is_key_passed("-p") ) {
				mmul.print_results();
			}
			return 0;
		}

	}


	// Shorthand to multiply non-square matrices (default blocksize)
	else if ( input.is_key_passed("-n") && 
		 	  input.is_key_passed("-k") &&
		 	  input.is_key_passed("-m") && 
			 !input.is_key_passed("-b") ) {

		// Parse input values
		std::vector<std::string> n_key_data = input.get_key_values("-n");
		std::vector<std::string> k_key_data = input.get_key_values("-k");
		std::vector<std::string> m_key_data = input.get_key_values("-m");

		// Check that all dimensions are is greater than defined block size
		if ( std::stoi(n_key_data[0]) < KERNEL_DEFAULT_BLOCK_SIZE ){
	
			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(n_key_data[0]), KERNEL_DEFAULT_BLOCK_SIZE );
			return -1;
		}

		else if ( std::stoi(k_key_data[0]) < KERNEL_DEFAULT_BLOCK_SIZE ){

			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(k_key_data[0]), KERNEL_DEFAULT_BLOCK_SIZE );
			return -1;
		}	
		else if ( std::stoi(m_key_data[0]) < KERNEL_DEFAULT_BLOCK_SIZE ){

			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(m_key_data[0]), KERNEL_DEFAULT_BLOCK_SIZE );
			return -1;
		}

		else {
			
			cl_mmul_demo mmul( 
				std::stoi(n_key_data[0]), 
				std::stoi(k_key_data[0]), 
				std::stoi(m_key_data[0]) );
			mmul.gpu_product();

			// If CPU flag is passed run product
			if ( input.is_key_passed("-cpu") ){
				mmul.cpu_product();
			}

			// Run equivalence test
			mmul.equivalence_test();

			// Print results if desired
			if ( input.is_key_passed("-p") ) {
				mmul.print_results();
			}
			return 0;
		}
	}

	// Shorthand to multiply non-square matrices (custom blocksize)
	else if ( input.is_key_passed("-n") && 
			  input.is_key_passed("-k") &&
			  input.is_key_passed("-m") && 
			  input.is_key_passed("-b") ) {

		std::vector<std::string> n_key_data = input.get_key_values("-n");
		std::vector<std::string> k_key_data = input.get_key_values("-k");
		std::vector<std::string> m_key_data = input.get_key_values("-m");
		std::vector<std::string> b_key_data = input.get_key_values("-b");


		// Check that all dimensions are is greater than defined block size
		if ( std::stoi(n_key_data[0]) < std::stoi(b_key_data[0]) ){
	
			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(n_key_data[0]), std::stoi(b_key_data[0]));
			return -1;
		}

		else if ( std::stoi(k_key_data[0]) < std::stoi(b_key_data[0]) ){

			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(k_key_data[0]), std::stoi(b_key_data[0]));
			return -1;
		}	
		else if ( std::stoi(m_key_data[0]) < std::stoi(b_key_data[0]) ){

			printf("error: size(%d) < block_size(%d)\n\n", std::stoi(m_key_data[0]), std::stoi(b_key_data[0]));
			return -1;
		}

		else {

			// Perform GPU multiplication
			cl_mmul_demo mmul( 
				std::stoi(n_key_data[0]), 
				std::stoi(k_key_data[0]), 
				std::stoi(m_key_data[0]),
				std::stoi(b_key_data[0]) );
			mmul.gpu_product();

			// If CPU flag is passed run product
			if ( input.is_key_passed("-cpu") ){
				mmul.cpu_product();
			}

			// Run equivalence test
			mmul.equivalence_test();

			// Print results if desired
			if ( input.is_key_passed("-p") ) {
				mmul.print_results();
			}
			return 0;
		}
	}

	else {

		printf("See -h for usage \n\n");
		return 0; 
	}
}