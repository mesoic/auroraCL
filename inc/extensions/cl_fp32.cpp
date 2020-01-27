// ---------------------------------------------------------------------------------
//	auroraCL -> inc/extenstions/fp32.cpp
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

template<class T>
void cl_matrix<T>::show_threads( 
	cl_device device, cl::NDRange gNDR, cl::NDRange lNDR, cl::NDRange lWPT ){

	// Cast this pointer as A
	cl_matrix<T> A = *this;

	// Try show_threads()
	try {

		// Retrieve Kernel
		cl::Kernel kernel = device.get_kernel("f32_show_threads"); 

		// Create command queue
		cl::CommandQueue queue(device.context, device.device);

		// Set kernel args
		kernel.setArg(0, (const int)A.m);
		kernel.setArg(1, (const int)A.n);
		kernel.setArg(2, (const int)lWPT[0]);
		kernel.setArg(3, (const int)lWPT[1]);

		// Transform indices for thread multi-tasking
		cl::NDRange _gNDR( gNDR[0] / lWPT[0], gNDR[1] / lWPT[1] );
		cl::NDRange _lNDR( lNDR[0] / lWPT[0], lNDR[1] / lWPT[1] );

		// Enque and run the kernel
		queue.enqueueNDRangeKernel( kernel, cl::NullRange, _gNDR, _lNDR );
	}

	// If exception is thrown it will be caught here
	catch (cl::Error& e) {
		printf("Runtime Error(%d): %s\n", e.err(), device.get_error_string( e.err() ) );
		printf("  what(): %s\n", e.what() );
		exit(1);
	}	
}

template<class T>
cl_matrix<T> cl_matrix<T>::product(
	cl_matrix<T> B, cl_device device, const char* kernel_name, cl::NDRange NDR ){

	// Cast this pointer as A
	cl_matrix<T> A = *this;
	cl_int Error = 0;
	
	// Check type equivalence
	if ( strcmp( A.m_type_t, B.m_type_t) != 0 ){
		std::cout<<"Buffer error: Conflicting types for matrices\n";
		std::cout<<"matrix(A) = "<<A.m_type_t<<"\n";
		std::cout<<"matrix(B) = "<<B.m_type_t<<"\n";
		return cl_matrix<T>(A.m, B.n);
	}

	// Check dimensions
	if ( A.n != B.m ){
		printf(
			"Unable to broadcast shapes %d(rows) x %d(cols) and %d(rows) x %d(cols)\n >> Returning zeros\n", 
			(int)A.m, 
			(int)A.n, 
			(int)B.m,
			(int)B.n
		);
		return cl_matrix<T>(A.m, B.n);
	}	

	// Check tiling
	// if ( ( (  A.m % NDR[0] != 0 ) || ( B.n % NDR[1] != 0 ) ) && 
	// 	 ( strcmp( kernel_name, "f32_product_v1" ) == 0 ) ) {
	// 	printf("Unalinged tiling (%d, %d) on output matrix %d(rows) x %d(cols)\n >> Returning zeros\n\n",
	// 		(int)NDR[0], 
	// 		(int)NDR[1],
	// 		(int)A.m,
	// 		(int)B.n
	// 	);
	// 	return cl_matrix<T>(A.m, B.n);
	// }


	// Exception handler for OpenCL calls
	try {

		// Create command queue
		cl::CommandQueue queue(device.context, device.device);

		// Create buffer objects for result matrix
		T *buffer = (T *)malloc(sizeof(B.m_size_t)*A.m*B.n);

		// Declare cl::Buffers for matrices
		cl::Buffer buffer_A; 
		cl::Buffer buffer_B; 
		cl::Buffer buffer_C; 

		// Allocate buffers. Implemented as pinned memory (zero copy)
		buffer_A = cl::Buffer(device.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  A.m_size_t*A.m*A.n, NULL, &Error);
		buffer_B = cl::Buffer(device.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  B.m_size_t*B.m*B.n, NULL, &Error);
		buffer_C = cl::Buffer(device.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, B.m_size_t*A.m*B.n, NULL, &Error);

		// non-blocking write to buffers
		queue.enqueueWriteBuffer(buffer_A, CL_FALSE, 0, A.m_size_t*A.m*A.n, &A.data[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_FALSE, 0, B.m_size_t*B.m*B.n, &B.data[0]);

		// Each matrix multiplicataion kernel requires different configuration of the API.
		// Kernel v0: Simple mmul w/global memory access (__global)  
		if (  strcmp (kernel_name, "f32_product_v0" ) == 0  ){

			// Retrieve Kernel
			cl::Kernel kernel = device.get_kernel(kernel_name); 

			// Set kernel args
			kernel.setArg(0, (const int)A.m);
			kernel.setArg(1, (const int)B.n);
			kernel.setArg(2, (const int)A.n);
			kernel.setArg(3, buffer_A);
			kernel.setArg(4, buffer_B);
			kernel.setArg(5, buffer_C);

			// Enqueue buffer write and kernel execute commands
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(A.m, B.n), NDR);

			// Blocking read of data into buffers
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, A.m_size_t*A.m*B.n, buffer);
			queue.finish();
		}


		// Kernel v1: mmul with local memory tiling (__local)
		if (  strcmp (kernel_name, "f32_product_v1" ) == 0  ){

			// Retrieve Kernel
			cl::Kernel kernel = device.get_kernel(kernel_name); 

			// Set kernel args
			kernel.setArg(0, (const int)A.m);
			kernel.setArg(1, (const int)B.n);
			kernel.setArg(2, (const int)A.n);
			kernel.setArg(3, buffer_A);
			kernel.setArg(4, buffer_B);
			kernel.setArg(5, buffer_C);
		 	kernel.setArg(6, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );
		 	kernel.setArg(7, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );

			// Enqueue buffer write and kernel execute commands
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(A.m, B.n), NDR);

			// Blocking read of data into buffers
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, A.m_size_t*A.m*B.n, buffer);
			queue.finish();
		}


		// Kernel v2: mmul with 1D-thread reduction (__private)
		if (  strcmp (kernel_name, "f32_product_v2" ) == 0  ){

			// Define work per thread
			const int wptN = NDR[1];

			// Calculate transformed NDRange(s) (__gloabl/__local)
			cl::NDRange G_NDR( A.m, B.n / wptN );
			cl::NDRange L_NDR( NDR[0], NDR[1] / wptN );

			// Retrieve Kernel
			cl::Kernel kernel = device.get_kernel(kernel_name); 

		 	// Set kernel args
		 	kernel.setArg(0, (const int)A.m);
		 	kernel.setArg(1, (const int)B.n);
		 	kernel.setArg(2, (const int)A.n);
		 	kernel.setArg(3, buffer_A);
		 	kernel.setArg(4, buffer_B);
		 	kernel.setArg(5, buffer_C);
		  	kernel.setArg(6, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );
		  	kernel.setArg(7, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );
		  	
			// Enqueue buffer write and kernel execute commands
			queue.enqueueNDRangeKernel( kernel, cl::NullRange, G_NDR, L_NDR );

			// Blocking read of data into buffers
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, A.m_size_t*A.m*B.n, buffer);
			queue.finish();
		}


		// Kernel v3: mmul with 2D-thread reduction (__private)
		// if (  strcmp (kernel_name, "f32_product_v3" ) == 0  ){

		// 	// Define work per thread
		// 	const int wptM = NDR[0];
		// 	const int wptN = NDR[1];

		// 	// Calculate transformed NDRange(s) (__gloabl/__local)
		// 	cl::NDRange G_NDR( A.m / wptM, B.n / wptN );
		// 	cl::NDRange L_NDR( NDR[0] / wptM, NDR[1] / wptN );

		// 	// Set kernel args
		// 	kernel.setArg(0, (const int)A.m);
		// 	kernel.setArg(1, (const int)B.n);
		// 	kernel.setArg(2, (const int)A.n);
		// 	kernel.setArg(3, buffer_A);
		// 	kernel.setArg(4, buffer_B);
		// 	kernel.setArg(5, buffer_C);
		// 	kernel.setArg(6, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );
		// 	kernel.setArg(7, cl::Local( NDR[0]*NDR[1]*A.m_size_t ) );
		// 	kernel.setArg(8, (const int)wptM);
		// 	kernel.setArg(9, (const int)wptN);

		// 	// Enqueue buffer write and kernel execute commands
		// 	queue.enqueueNDRangeKernel( kernel, cl::NullRange, G_NDR, L_NDR );

		// 	// Blocking read of data into buffers
		// 	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, A.m_size_t*A.m*B.n, buffer);
		// 	queue.finish();
		// }


		// Create a new matrix and copy in data
		cl_matrix<T> C(A.m, B.n, buffer);
		free(buffer);

		// Return matrix
		return C;
	}

	// If exception is thrown it will be caught here
	catch (cl::Error& e) {
		printf("Runtime Error(%d): %s\n", e.err(), device.get_error_string( e.err() ) );
		printf("  what(): %s\n", e.what() );
		exit(1);
	}	
}
