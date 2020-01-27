// ---------------------------------------------------------------------------------
//	auroraCL -> kernels/f32/cl_research.cpp
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
// AuroraCL: Research Kernels
//
// f32_product_v2: 2D register tiling (i.e. '2D more work per thread')
//
// Issues: Kernel forms mathematically accurate products. Yet compute time
// is slow. Consider alternative hardware test, memory investigations and
// compiler optimization.
__kernel void f32_product_v3(
	const int M, 
	const int N, 
	const int K, 
	__global float *A, 
	__global float *B, 
	__global float *C,
	__local float *Asub,
	__local float *Bsub, 
	const int WORK_PER_THREAD_M,
	const int WORK_PER_THREAD_N )
{

	__constant int WPTM = 8; 
	__constant int WPTN = 8;

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
	const int N_TILES = K / WPTN;

	// Allocate registers 
	__private float Breg;
	__private float Areg[ 8 ];
	__private float  acc[ 8 * 8 ];

	// Initialize accumulation buffers
	for (int IT = 0; IT < WPTM * WPTN; IT++){ acc[ IT ] = 0.0f; }

	// Calculate number of tiles
	for ( int tile = 0; tile < N_TILES; tile++){

		// Horizontal offset
		int TILE_OFFSET = (tile)*(TILE_SIZE_N)*(WPTN);
	
		// Load one Atile and one Btile into __local 
		for ( int wM = 0; wM < WPTM; wM++) {

			for ( int wN = 0; wN < WPTN; wN++) {

				// Local memory permuatator
				int lINDEX = ( wM * WPTN ) + ( wN );

				// Corresponding global index expressed as (row)*K + (col)
				int aINDEX = ( ( GLOBAL_M * WPTN + wM ) * K ) + ( TILE_OFFSET + wN );
				int bINDEX = ( ( TILE_OFFSET + wM ) * N ) + ( GLOBAL_N * WPTN + wN );

				// Store values in local memory
				Asub[ lINDEX ] = A[ aINDEX ];
				Bsub[ lINDEX ] = B[ bINDEX ];
			}
		}

		// Synchronization barrier (load)
		barrier(CLK_LOCAL_MEM_FENCE);

		for ( int wM = 0; wM < WPTM ; wM++) {

			// Preload row of the Asub into registers				
			for ( int regN = 0; regN < WPTN; regN++) {
				int regINDEX = ( WPTN * wM ) + regN;
				Areg[ regN ] = Asub[ regINDEX ];
			}

			// Calculate the product
			for ( int wN = 0; wN < WPTN; wN++){

				// Assign accumulation buffer index
				int accINDEX = ( wM * WPTN ) + wN;

				// Loop through the Bsub matrix to get col wN
				for ( int regM = 0; regM < WPTM; regM++){

					int regINDEX = ( WPTM * regM ) + wN; 
					Breg = Bsub[ regINDEX ];
					acc[ accINDEX ] += Areg[ regM ]*Breg;
				}
			}
		}

		// Synchronization barrier (product)	
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store values
	for ( int wM = 0; wM < WPTM; wM++) {
		for ( int wN = 0; wN < WPTN; wN++) {
			
			int gINDEX 	 = ( ( GLOBAL_M * WPTM + wM ) * N ) + ( GLOBAL_N * WPTN + wN );
			int accINDEX = ( wM * WPTN ) + wN;
			C[ gINDEX ]  = acc[ accINDEX ];
	
		}
	}
}