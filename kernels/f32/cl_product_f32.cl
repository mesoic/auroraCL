// ---------------------------------------------------------------------------------
//	auroraCL -> kernels/f32/cl_product.cpp
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

//
// AuroraCL Kernels for Matrix Product (f32). 
//
// Kernel Status:
// f32_product_v0: Confirmed
// f32_product_v1: Confirmed
// f32_product_v2: Confirmed
//
// matrix_a = m(rows) x k(cols)
// matrix_b = k(rows) x n(cols)
// matrix_c = m(rows) x n(cols)
//
// f32_product_v0: naive algorithm 
__kernel void f32_product_v0 ( 
	const int M,
	const int N,
	const int K,
	__global float *A,
	__global float *B,
	__global float *C )

{
	// Thread identifiers (__global)
	const int GLOBAL_M = get_global_id(0);
	const int GLOBAL_N = get_global_id(1); 

	// Global identifier
	const int gINDEX = ( GLOBAL_M * N ) + GLOBAL_N;

	// Allocate accumulation buffer
	float acc = 0.0f;

	// Perform the calculation
	for ( int IT = 0; IT < K; IT++ ) {

		// Calculation of aINDEX/bINDEX
		int aINDEX = ( GLOBAL_M * K ) + ( IT );
		int bINDEX = ( IT * N ) + ( GLOBAL_N );
		acc += A[ aINDEX ] * B[ bINDEX ];
	}
	
	// Store result
	C[ gINDEX ] = acc;	
	#pragma PKP QED
} 

// f32_product_v1: leveraging local memory
__kernel void f32_product_v1 (
		const int M, 
		const int N, 
		const int K, 
		__global float *A, 
		__global float *B, 
		__global float *C,
		__local float *Asub,
		__local float *Bsub )

{
	// Thread identifiers (__global)
	const int GLOBAL_M = get_global_id(0);
	const int GLOBAL_N = get_global_id(1);
	
	// Thread identifiers (__local)
	const int LOCAL_M = get_local_id(0); 
	const int LOCAL_N = get_local_id(1); 

	// Number of threads(__local) 
	const int LOCAL_SIZE_M = get_local_size(0);
	const int LOCAL_SIZE_N = get_local_size(1);

	// __global(__local) indices (vector valued)
	const int gINDEX = ( GLOBAL_M * N ) + GLOBAL_N;
	const int lINDEX = ( LOCAL_M * LOCAL_SIZE_N ) + LOCAL_N;

	// Define tile sizes and calculate the number of tiles
	const int TILE_SIZE_M = LOCAL_SIZE_M;
	const int TILE_SIZE_N = LOCAL_SIZE_N; 
	const int N_TILES = K / TILE_SIZE_N;

	// Initialize accumulation buffer
	float acc = 0.0f;

	// Perform the calculation
	for ( int tile = 0; tile < N_TILES; tile++ ){

		// Offset variable
		int TILE_OFFSET = (tile)*TILE_SIZE_N;
	
		// Calculation of aINDEX/bINDEX
		int aINDEX = ( GLOBAL_M * K ) + ( TILE_OFFSET + LOCAL_N );
		int bINDEX = ( ( TILE_OFFSET + LOCAL_M ) * N ) + ( GLOBAL_N );	

		// Copy submatrices into local memory
		Asub[ lINDEX ] = A[ aINDEX ];
		Bsub[ lINDEX ] = B[ bINDEX ];

		// Synchronization barrier (load)
		barrier( CLK_LOCAL_MEM_FENCE );

		// Multiply submatrices
		for ( int IT = 0; IT < TILE_SIZE_N; IT++ ){	 
			int asubINDEX = ( LOCAL_M * TILE_SIZE_N ) + ( IT );
			int bsubINDEX = ( TILE_SIZE_N * IT ) + ( LOCAL_N );
			acc += Asub[ asubINDEX ] * Bsub[ bsubINDEX ];
		}

		// Synchronization barrier (product)
		barrier( CLK_LOCAL_MEM_FENCE );
	}
	
	// Store result
	C[ gINDEX ] = acc;
	#pragma PKP QED
}

// f32_product_v2: 1D register tiling (i.e. 'more work per thread')
__kernel void f32_product_v2 (
		const int M, 
		const int N, 
		const int K, 
		__global float *A, 
		__global float *B, 
		__global float *C,
		__local float *Asub,
		__local float *Bsub )

{
	// Kernel Preprocessor
	#pragma PKP WORK_PER_THREAD_N __default 8
	#ifndef WORK_PER_THREAD_N
		#define WORK_PER_THREAD_N 8
	#endif

	// Preprocessor definitions
	__constant int WPTN = WORK_PER_THREAD_N;

	// Thread identifiers (__global)
	const int GLOBAL_M = get_global_id(0);
	const int GLOBAL_N = get_global_id(1);
	
	// Thread identifiers (__local)
	const int LOCAL_M = get_local_id(0);
	const int LOCAL_N = get_local_id(1);

	// Number of threads(__local) 
	const int LOCAL_SIZE_M = get_local_size(0);
	const int LOCAL_SIZE_N = get_local_size(1);

	// Calculate number of tiles
	const int TILE_SIZE_N = LOCAL_SIZE_N; 
	const int N_TILES = get_num_groups(0);

	// Initialize Aregister and accumulation buffer
	__private float Areg;
	__private float acc[ WPTN ];
	for (int wN = 0; wN < WPTN; wN++){ acc[wN] = 0.0f; }

	// Perform the calculation
	for ( int tile = 0; tile < N_TILES; tile++ ){

		// Offset variable
		int TILE_OFFSET = (tile)*(TILE_SIZE_N)*(WPTN);

		// Loop through sub-columns
		for ( int wN = 0; wN < WPTN; wN++ ){

			// Local memory permuatator
			int lINDEX = ( LOCAL_M * LOCAL_SIZE_M ) + ( LOCAL_N * WPTN + wN );

			// Corresponding global index expressed as (row)*K + (col)
			int aINDEX = ( ( GLOBAL_M ) * K ) + ( TILE_OFFSET + wN );
			int bINDEX = ( ( TILE_OFFSET + LOCAL_M ) * N ) + ( GLOBAL_N * WPTN  + wN );

			// Store values in local memory
			Asub[ lINDEX ] = A[ aINDEX ];
			Bsub[ lINDEX ] = B[ bINDEX ];
		}

		// Synchronization barrier (load)
		barrier(CLK_LOCAL_MEM_FENCE);

		// Multiply submatrices
		for( int IT = 0; IT < (TILE_SIZE_N)*(WPTN); IT++ ) {
			
		 	// Calculate aINDEX and strore in register
			int asubINDEX = ( LOCAL_M * WPTN ) + ( LOCAL_N + IT );
		 	Areg = Asub[ asubINDEX ];

		 	for ( int wN = 0; wN < WPTN; wN++ ){
				
		 		int bsubINDEX = ( LOCAL_N + WPTN * IT ) + wN;
		 		acc[ wN ] += Areg * Bsub[ bsubINDEX ];
		 	} 
		}

		// Synchronization barrier (product)
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the result	
	for (int wN = 0; wN < WPTN; wN++ ) {
		int gINDEX = ( GLOBAL_M * N ) + ( GLOBAL_N * WPTN + wN );
		C[ gINDEX ] = acc[ wN ];
	}
	#pragma PKP QED
}