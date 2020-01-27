// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_time.cpp
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
#include <chrono>
#include <thread>

class cl_time {

	public:

		// cl_time will express floating point microseconds 
		using cl_time_t = std::chrono::duration<float, std::micro>;
		std::chrono::time_point<std::chrono::steady_clock> t0, t1;

		// Variable for timedelta
		cl_time_t d0;

		// constructor
		cl_time(void);
		~cl_time(void);

		// start and end methods
		void start(void);
		void end(void);

		// Calculation of timedelta
		cl_time_t delta(void);

		// Print result
		void print(void);
		void print(cl_time_t);

		// Sleep ms function
		void sleep_ms(int t);
		void sleep_us(int t);

};

// Constructor initialize t0 and t1 as current timepoint
cl_time::cl_time(void) { 
	this->t0 = std::chrono::steady_clock::now();
	this->t1 = this->t0;
	this->d0 = this->t1 - this->t0;
}

cl_time::~cl_time(void) { }

// Methods to evaluate timepoints and timedelta
inline void cl_time::start(void){ 
	this->t0 = std::chrono::steady_clock::now(); 
}

inline void cl_time::end(void){ 
	this->t1 = std::chrono::steady_clock::now(); 
	this->d0 = this->t1 - this->t0;
}

inline cl_time::cl_time_t cl_time::delta(){ 
	return this->d0;
}

// Method to print the timedelta
inline void cl_time::print(void){
	std::cout<<"Elapsed Time: "<<(this->d0).count()<<"us"<<std::endl;
}

// Sleep functions
inline void cl_time::sleep_ms(int t){
	std::this_thread::sleep_for(std::chrono::milliseconds(t));
}

inline void cl_time::sleep_us(int t){
	std::this_thread::sleep_for(std::chrono::microseconds(t));
}