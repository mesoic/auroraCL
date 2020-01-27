// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_file.cpp
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
#include <fstream>
#include <string>
#include <vector>

// AuroraCL packages
#include "../inc/cl_matrix.hpp"

// JSON decode (parser:JSON for modern cpp)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class cl_file {

	public: 
		// Data members
		const char* filename;
		
		// File pointer
		std::fstream f;

		// Constructor
		cl_file(const char* filename); 
		~cl_file(void);

		// Define ascii deliminators
		char delim_c[1] = {','};
		char delim_r[1] = {':'};

		// Define ascii r/w methods
		template<class T>
		void write_ascii( cl_matrix<T>& A, const char* key, const char* mode = 'w' );
		
		template<class T>
		cl_matrix<T> read_ascii( const char* key );
};

// Constructor
cl_file::cl_file(const char* filename){ this->filename = filename;  }

// Destructor
cl_file::~cl_file(void){}

// Write matrix in ascii format
template<class T> 
void cl_file::write_ascii( cl_matrix<T>& data, const char* key, const char* mode ){
	
	if ( strcmp( mode, "w" ) == 0 )
		this->f.open (this->filename, std::fstream::out);
	if ( strcmp( mode, "w+" ) == 0 )	
	  	this->f.open (this->filename, std::fstream::out | std::fstream::app);

	if ( this->f.is_open() ){
		this->f<<"matrix:"<<key<<std::endl;

		for ( size_t i = 0; i < data.m; i++){
			for ( size_t j = 0; j < data.n; j++){
				char d[1] = { (j == data.n-1) ? this->delim_r[0] :this->delim_c[0] };
				this->f<<data.get_elem(i,j)<<d;
			}
			this->f<<std::endl;
		}

		this->f<<"end:"<<std::endl;
		this->f.close();
	}
}

// Read matrix in ascii format 
template<class T> 
cl_matrix<T> cl_file::read_ascii(const char* key){

	// Open ascii file
	this->f.open (this->filename, std::fstream::in );
	std::string line;

	// Initialize vector to buffer data
	std::vector<std::vector<T>> buffer;

	// Key exists
	bool key_exists = false;

	// Parse the file
	if ( this->f.is_open() ) {

		// Loop through file
		while( std::getline(this->f, line) ){

			char* token_k = strtok( &line[0], delim_r); 

			// Layer1: Check if line begins a matrix
			while( token_k ) {
		
				// Check for matrix begin and store key
				if ( strcmp(token_k, "matrix") == 0 ){
 
 					// token now contains key
					token_k = strtok(NULL, delim_r); 
					if ( strcmp(token_k, key)  == 0 ){

						key_exists = true;

						// Layer2: Loop through matrix file
						while( std::getline(this->f, line) ){

							// Check for matrix end and store token 
							token_k = strtok( &line[0], delim_r );
					
							if ( strcmp(token_k, "end") == 0 )
								break;
						
							// token_k now contains matrix data
							else{
								std::vector<T> row;
								char* token_c = strtok( &token_k[0], delim_c );
								while( token_c ){
									row.push_back( (T)std::stof( token_c ) );
									token_c = strtok(NULL, delim_c); 
								}
								buffer.push_back(row); // push row
							}
						}
					}
				}
				token_k = strtok(NULL, delim_r);
			}
		}
		// close file pointer
		this->f.close();
	}

	// Assert that key has been found
	if( !key_exists ){
		printf("Key Error: matrix:%s not found in file %s\n", key, filename);
		exit(1);
	}
 
	// Buffer is now <std::vector<std::vector<T>>
	size_t m = buffer.size();
	size_t n = buffer[0].size(); 

	// Initialize matrix and read in rows
	cl_matrix<T> _(m,n);
	for ( size_t i = 0; i<m; i++)
		_.update_row( i, buffer[i] );


	// return data
	return _;
}