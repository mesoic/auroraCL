// ---------------------------------------------------------------------------------
//	auroraCL -> kernels/f32/cl_utils.cpp
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
// AuroraCL Utility Kernels (f32). 
//
// matrix_a = m(rows) x k(cols)
// matrix_b = k(rows) x n(cols)
// matrix_c = m(rows) x n(cols)
//
__kernel void f32_show_threads( 
	const int M, 
	const int N, 
	const int WORK_PER_THREAD_M, 
	const int WORK_PER_THREAD_N )

{

	// Number of threads on domain (__global)
	const int GLOBAL_SIZE_M = get_global_size(0); // Global dim(x)
	const int GLOBAL_SIZE_N = get_global_size(1); // Global dim(y)

	// Number of thread blocks on domain (__global)
	const int BLOCK_SIZE_M = get_num_groups(0); // Work group dim(x)
	const int BLOCK_SIZE_N = get_num_groups(1); // Work group dim(y)

	// Number of threads per thread block (__local)
	const int LOCAL_SIZE_M = get_local_size(0); // Work group dim(x)
	const int LOCAL_SIZE_N = get_local_size(1); // Work group dim(y)

	// Thread identifiers (__global)
	const int GLOBAL_M = get_global_id(0); // Global ID(X) 
	const int GLOBAL_N = get_global_id(1); // Global ID(y)

	// Thread block identifiers (__global)
	const int BLOCK_M	= get_group_id(0); // Work group ID(x)
	const int BLOCK_N	= get_group_id(1); // Work group ID(y)

	// Thread identifiers (__local)
	const int LOCAL_M = get_local_id(0); // Work ITem ID(x)
	const int LOCAL_N = get_local_id(1); // Work ITem ID(y)

	// Global vector valued index
	const int gINDEX = ( ( GLOBAL_M * WORK_PER_THREAD_M ) * N ) + ( GLOBAL_N * WORK_PER_THREAD_N ) ;

	// Show global data
	if (GLOBAL_M == 0 && GLOBAL_N == 0){
		printf(
			"(matrix)\n\t| g_dim(%d, %d)\n\t| b_dim(%d, %d)\n\t| l_dim(%d, %d)\n\t| l_wpt(%d, %d)\n", 
			GLOBAL_SIZE_M, GLOBAL_SIZE_N, 
			BLOCK_SIZE_M, BLOCK_SIZE_N, 
			LOCAL_SIZE_M, LOCAL_SIZE_N, 
			WORK_PER_THREAD_M, WORK_PER_THREAD_N
		);
	}

	// If performing register tiling, populate 'responsible for' data
	printf("(index = %d)\n\t| g(%d, %d)\n\t| b(%d, %d)\n\t| l(%d, %d)\n", 
		gINDEX, GLOBAL_M, GLOBAL_N, BLOCK_M, BLOCK_N, LOCAL_M, LOCAL_N );

}