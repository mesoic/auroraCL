// ---------------------------------------------------------------------------------
//	auroraCL -> inc/cl_matrix.cpp
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

#include <vector>
#include <random>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <iostream>

// Class defining cl_matrix type
template <class T>
class cl_matrix {

	public: 
	
		size_t m;	// m-rows 
		size_t n;	// n-cols

		std::vector<T> data; // matrix as multi-indexable vector
			
		const char* m_type_t; 	// type
		size_t m_size_t;  		// size of type <T> for GPU malloc
		
		// Constructor
		cl_matrix(size_t m, size_t n, const bool identity = false);
		cl_matrix(size_t m, size_t n, T* buffer);
		cl_matrix(void);
		~cl_matrix(void);

		// Setter/Getter methods (elementwise)
		T get_elem(size_t i, size_t j);
		void set_elem(size_t i, size_t j, T val);

		// Update methods
		void update_row(size_t k, std::vector<T> data);
		void update_col(size_t k, std::vector<T> data);

		// Fill rand method
		void fill_rand(T a, T b, T norm = 1.0);
		void fill_ints();

		// Exchange methods
		cl_matrix<T> exchange_row(size_t k, cl_matrix<T> A);
		cl_matrix<T> exchange_col(size_t k, cl_matrix<T> A);

		// Swap Methods
		cl_matrix<T> swap_row(size_t m1, size_t m2);
		cl_matrix<T> swap_col(size_t n1, size_t n2);

		// Operator overloads and dot 
		cl_matrix<T> operator+(cl_matrix<T> A);
		cl_matrix<T> operator-(cl_matrix<T> A);
		cl_matrix<T> operator*(cl_matrix<T> A);	
		cl_matrix<T> dot(cl_matrix<T> A);

		// Scalar multiplication
		cl_matrix<T> operator*(T val);
		
		bool operator==(cl_matrix<T> A);
		bool operator!=(cl_matrix<T> A);
		cl_matrix<T> operator+=(cl_matrix<T> A);
		cl_matrix<T> operator-=(cl_matrix<T> A);

		// Matrix specific calculations 
		cl_matrix<T> product(cl_matrix<T> A);
		cl_matrix<T> transpose(void);
		cl_matrix<T> inv(void);

		// Determinant and trace
		T det(void);
		T tr(void);

		// Is vector and is_square methods
		bool is_square(void);
		bool is_vector(void);
		
		// Print methods
		void config(void);
		void pprint(const char* str = "\0");

		// Method to extract datatypes
		void show_types(void);

		// GPU Implementations
		void show_threads(
			cl_device device, 
			cl::NDRange gNDR, 
			cl::NDRange lNDR, 
			cl::NDRange lWPT = cl::NDRange(1,1)
		);
 
 		// Product function
		cl_matrix<T> product(
			cl_matrix<T> A, 
			cl_device device, 
			const char* kernel_name = "cl_product_v0",
			cl::NDRange NDR = cl::NDRange(8,8)
		);

};

// Constructor
template<class T>
cl_matrix<T>::cl_matrix(size_t m, size_t n, const bool identity){

	// Matrix dimensions 
	this->m = m; 
	this->n = n;

	// Call to pretty function compiler macro for typestring
	this->m_type_t = __PRETTY_FUNCTION__;	
	this->m_size_t = sizeof(T);

	// Call std::vector<T> constructor
	this->data = std::vector<T>(this->m*this->n);

	// If initialized as identity matrix
	if ( identity )
		for ( size_t i = 0; i < this->m; i++ )
			this->set_elem(i,i,1);
}	

// Constructor from data
template<class T>
cl_matrix<T>::cl_matrix(size_t m, size_t n, T* buffer){

	// Matrix dimensions 
	this->m = m;
	this->n = n;

	// Call to pretty function compiler macro for typestring
	this->m_type_t = __PRETTY_FUNCTION__;	
	this->m_size_t = sizeof(T);

	// Calculate offset and initialize data
	this->data = std::vector<T>(buffer, buffer + ( this->m * this->n ) );
}

// Null constructor 
template<class T>
cl_matrix<T>::cl_matrix() {}

// Destructor
template<class T>
cl_matrix<T>::~cl_matrix() {}

// Get element method
template<class T>
T cl_matrix<T>::get_elem(size_t i, size_t j){ return this->data[i*this->n + j]; }

// Set element method
template<class T>
void cl_matrix<T>::set_elem(size_t i, size_t j, T val){ this->data[i*this->n + j] = val; }

// Update row
template<class T>
void cl_matrix<T>::update_row(size_t k, std::vector<T> v){

	if(v.size() != this->n)  {
		printf(
			"Unable to broadcast row(len=%d) into matrix %d(rows) x %d(cols)\n", 
			(int)v.size(), 
			(int)this->m, 
			(int)this->n
		);
		exit(1);
	}
	else {
		for (size_t i = 0; i<v.size(); i++)
			this->set_elem(k, i, v[i]);
	}
}

// Update col
template<class T>
void cl_matrix<T>::update_col(size_t k, std::vector<T> v){

	if(v.size() != this->m)  {
		printf(
			"Unable to broadcast col(len=%d) into matrix %d(rows) x %d(cols)\n", 
			(int)v.size(), 
			(int)this->m, 
			(int)this->n
		);
		exit(1);
	}
	else {
		for (size_t i = 0; i<v.size(); i++)
			this->set_elem(i, k, v[i]);
	}
}

// Exchange row
template<class T>
cl_matrix<T> cl_matrix<T>::exchange_row(size_t k, cl_matrix<T> A){

	cl_matrix<T> C(A.m, A.n);
	C.data = this->data;

	for (size_t j=0; j<this->n; j++){
		C.set_elem(k,j, A.get_elem(k,j) ); 
	}
	return C;
}

// Exchange col
template<class T>
cl_matrix<T> cl_matrix<T>::exchange_col(size_t k, cl_matrix<T> A){

	cl_matrix<T> C(A.m, A.n);
	C.data = this->data;

	for (size_t j=0; j<this->n; j++){
		C.set_elem(j,k, A.get_elem(j,k) ); 
	}
	return C;
}

// Swap row
template<class T>
cl_matrix<T> cl_matrix<T>::swap_row(size_t m1, size_t m2){

	cl_matrix<T> C(this->m, this->n);
	C.data = this->data;

	for (size_t j=0; j<this->n; j++){
		C.set_elem(m1, j, this->get_elem(m2, j) );
		C.set_elem(m2, j, this->get_elem(m1, j) );
	}
	return C;
}

// Swap col
template<class T>
cl_matrix<T> cl_matrix<T>::swap_col(size_t m1, size_t m2){

	cl_matrix<T> C(this->m, this->n);
	C.data = this->data;

	for (size_t j=0; j<this->m; j++){
		C.set_elem(j, m1, this->get_elem(j, m2) );
		C.set_elem(j, m2, this->get_elem(j, m1) );
	}
	return C;
}

// Fill random method
template<class T>
void cl_matrix<T>::fill_rand(T a, T b, T norm){

	std::random_device rd;  // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr((int)a, (int)b); // define the range

	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			this->set_elem(i,j, (T)distr(eng)/norm);
		}
	}
}

// Fill unique
template<class T>
void cl_matrix<T>::fill_ints( void ){
	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			this->set_elem(i,j, i*this->n + j );
		}
	}
}


// Operator Overloads (+/+=)
// += version returns '*this' whereas returns 'C'
template<class T>
cl_matrix<T> cl_matrix<T>::operator+(cl_matrix<T> A){

	cl_matrix<T> C(A.m, A.n);

	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			C.set_elem( i, j, this->get_elem(i,j) + A.get_elem(i,j) );
		}
	}
	return C;
}

template<class T>
cl_matrix<T> cl_matrix<T>::operator+=(cl_matrix<T> A){ 

	for (int i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			this->set_elem( i, j, this->get_elem(i,j) + A.get_elem(i,j) );
		}
	}
	return *this;
}

// Operator Overloads (-/-=)
// -= version returns '*this' whereas returns 'C'
template<class T>
cl_matrix<T> cl_matrix<T>::operator-(cl_matrix<T> A){

	cl_matrix<T> C(A.m, A.n);

	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			C.set_elem( i, j, this->get_elem(i,j) - A.get_elem(i,j) );
		}
	}
	return C;
}

template<class T>
cl_matrix<T> cl_matrix<T>::operator-=(cl_matrix<T> A){ 

	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			this->set_elem( i, j, this->get_elem(i,j) - A.get_elem(i,j) );
		}
	}
	return *this;
}

// Dot method (elementwise multiplicaton)
template<class T>
cl_matrix<T> cl_matrix<T>::dot(cl_matrix<T> A){

	cl_matrix<T> C(A.m, A.n);

	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			C.set_elem(i, j , this->get_elem(i,j)*A.get_elem(i,j) );
		}
	}
	return C;
}

// Matrix multiplication
// Note that we will also overload operator*
template<class T>
cl_matrix<T> cl_matrix<T>::product(cl_matrix<T> A){

	// Check dimensions
	if (this->n != A.m){
		printf(
			"Unable to broadcast shape %d(rows) x %d(cols) into %d(rows) x %d(cols)\n", 
			(int)this->m, 
			(int)this->n, 
			(int)A.m,
			(int)A.n
		);
		exit(1);
	}

	// Define sizes
	size_t m = this->m;
	size_t K = this->n;
	size_t n = A.n;

	// Result and accumulation buffer
	cl_matrix<T> C(m, n);
	T acc;

	// Perform multiplication
	for (size_t i=0; i<m; i++){
		for (size_t j=0; j<n; j++){
			acc = 0;
			for (size_t k=0; k<K; k++) { 
				acc += this->get_elem(i, k) * A.get_elem(k, j);
			}
			C.set_elem(i,j, acc);
		}		
	}
	return C;
}

// Operator overload matrix multiplication (*)
template<class T>
cl_matrix<T> cl_matrix<T>::operator*(cl_matrix<T> A){ 
	return this->product(A); 
}

// Operator overload scalar multiplication (*)
template<class T>
cl_matrix<T> cl_matrix<T>::operator*(T val){

	cl_matrix<T> C(m, n);

	for (size_t i=0; i<m; i++){
		for (size_t j=0; j<n; j++){
			C.set_elem(i, j, val*this->get_elem(i,j) );
		}
	}

	return C;
}

// Operator Overloads (==/!=)
template<class T>
bool cl_matrix<T>::operator==(cl_matrix<T> A){

	// Check matrix dimenstions
	if (A.m != this->m || A.n != this->n){
		return false;
	}

	// Check matrix elements
	for (size_t i=0; i<this->m; i++){
		for (size_t j=0; j<this->n; j++){
			if( this->get_elem(i,j) != A.get_elem(i,j) ){
				return false;
			}	
		}
	}
	return true;
}

template<class T>
bool cl_matrix<T>::operator!=(cl_matrix<T> A){
	return !this->operator==(A);
}

// Matrix determinant via LU-decomposition O(n^3)
template<class T>
T cl_matrix<T>::det(void){

	// Assert matrix is square
	assert( this->is_square() );

	cl_matrix<T> L(this->m, this->n);
	cl_matrix<T> U(this->m, this->n);

	// Accumulation buffer
	// det() = Product of diagonal elements of U 
	T det = 1;
	for (size_t i = 0; i < this->n; i++) {
			
		for (size_t k = i; k < this->n; k++) {

			// Accumulation buffers
			T Uacc = 0, Lacc = 0;
			for (size_t j = 0; j < i; j++){
				Uacc += ( L.get_elem(i,j) * U.get_elem(j,k) );
				Lacc += ( L.get_elem(k,j) * U.get_elem(j,i) );
			}
			U.set_elem(i,k, (this->get_elem(i,k) - Uacc));
			L.set_elem(k,i, (this->get_elem(k,i) - Lacc)/U.get_elem(i,i) );
		}
		det*=U.get_elem(i,i);
	}
	return det;
}

// Matrix Inverse via LU-decomposition O(n^3)
template<class T>
cl_matrix<T> cl_matrix<T>::inv(void){

	// Assert matrix is square
	assert( this->is_square() );

	// Phase 1) Factoring A into LU-matrices
	// Note LU-symmetry in factoring and inversion routines
	cl_matrix<T> L(this->m, this->n);
	cl_matrix<T> U(this->m, this->n);

	for (size_t i = 0; i < this->n; i++) {
		
		for (size_t k = i; k < this->n; k++) {
	 
			// Accumulation buffers
			T Uacc = 0, Lacc = 0;

			// Summation of L(i, j) * U(j, k)
			// Summation of L(k, j) * U(j, i)
			for (size_t j = 0; j < i; j++){
				Uacc += ( L.get_elem(i,j) * U.get_elem(j,k) );
				Lacc += ( L.get_elem(k,j) * U.get_elem(j,i) );
			}

			// Evaluate U(i,k)/L(k,i)
			U.set_elem(i,k, (this->get_elem(i,k) - Uacc));
			L.set_elem(k,i, (this->get_elem(k,i) - Lacc)/U.get_elem(i,i) );
		}
	}

	// Phase 2) Inverting the LU-matrices
	cl_matrix<T> l(this->m, this->n);
	cl_matrix<T> u(this->m, this->n);

	// Note that we can invert the U matrix by exchanging indices
	for (size_t i = 0; i<this->n; i++){

		l.set_elem(i, i, 1.0/L.get_elem(i,i) ); // Sets diagonal element [i=0]
		u.set_elem(i, i, 1.0/U.get_elem(i,i) );	

		// Summation of u(i, j) * U(j, k) 
		// Summation of L(k, j) * l(j, i)
		for ( size_t k = i+1; k<this->n; k++){ 	 
			
			// Accumulatkon Buffers
			T Uacc = 0, Lacc = 0;
			for ( size_t j=0; j<k; j++){
				Uacc += -1*( u.get_elem(i,j)*U.get_elem(j,k) );
				Lacc += -1*( L.get_elem(k,j)*l.get_elem(j,i) );
			} 

			// Evaluate u(k,i)/l(i,k)
			u.set_elem(i,k, Uacc/U.get_elem(k,k));
			l.set_elem(k,i, Lacc/L.get_elem(k,k));
		}
	}

	// Phase 3) Taking the ul-product
	return u.product(l);
}

template<class T>
cl_matrix<T> cl_matrix<T>::transpose(void){

	cl_matrix<T> C(this->n, this->m);
	for (size_t i=0; i<this->n; i++){
		for (size_t j=0; j<this->m; j++){
			C.set_elem( i, j, this->get_elem(j,i) );
		}
	}
	return C; 
}

// Matrix trace 
template<class T>
T cl_matrix<T>::tr(void){

	T tr = 0;
	if (this->is_square()){ 
		for (size_t i=0; i<this->n; i++){ tr+=this->get_elem(i,i); } 
	}
	else { 
		std::cout<<"Warning: tr() undefined for non-square matrices"<<std::endl; 
	} 
	return tr;
}

// Print matrix methods
template<class T> 
void cl_matrix<T>::show_types(void){ 
	printf("Template Types:\n\t%s\n\tm_size_t = %d\n\n",this->m_type_t,(int)this->m_size_t); 
}

// v2) Prints matrix and prepends string
template<class T>
void cl_matrix<T>::pprint(const char* str){

	printf("%s[[\n",str);
	for (size_t i=0; i<this->m; i++) {
		for (size_t j=0; j<this->n; j++ ){
			std::cout << " " << this->get_elem(i,j);
		}
		printf("\n");
	}
	printf("]]\n\n");
}

// Test properties of the matrix
template<class T> inline bool cl_matrix<T>::is_square(void){ return (this->m == this->n) ? true : false; }
template<class T> inline bool cl_matrix<T>::is_vector(void){ return (this->m == 1 || this->n == 1) ? true : false;}

// Scalar multiplication (lexers)
// Would like 2*A and A*2 to behave as expected
template<class T> inline cl_matrix<T> operator*( cl_matrix<T> A, int val){return A.operator*( (T)val );}
template<class T> inline cl_matrix<T> operator*( cl_matrix<T> A, float val){return A.operator*( (T)val );}
template<class T> inline cl_matrix<T> operator*( cl_matrix<T> A, double val){return A.operator*( (T)val );}
template<class T> inline cl_matrix<T> operator*( int val, cl_matrix<T> A){ return A.operator*( (T)val ); }
template<class T> inline cl_matrix<T> operator*( float val, cl_matrix<T> A){ return A.operator*( (T)val ); }
template<class T> inline cl_matrix<T> operator*( double val, cl_matrix<T> A){ return A.operator*( (T)val ); }

// Include OpenCL function overloads
#include  "./extensions/cl_fp32.cpp"