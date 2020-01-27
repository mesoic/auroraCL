// ---------------------------------------------------------------------------------
//	auroraCL -> lib/interface/cl_parse.cpp
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

// For sanitize functions 
#include <algorithm>
#include <regex>

typedef bool (*function)(std::string key, std::vector<std::string>, std::vector<std::string>);

// This function which checks if the list of arguments 
// after a given key certain key is an integer.
bool sanitize_int(std::string key, std::vector<std::string> input_for_key, std::vector<std::string> input_key_vals ){


	// Check if there is only one string in the input
	if ( input_for_key.size() != 1 ){
		printf("Parse Error: Value for key (%s) must be an integer\n", 
			key.c_str() );
		return false;	
	}

	else {
	
		// Regex check for possible int
		if( std::regex_search( input_for_key[0], std::regex("\\d+") ) ) {
			return true;
		}	
		else {
			printf("Parse Error: Value (%s) for key (%s) must be an integer (representation)\n", 
				input_for_key[0].c_str(), key.c_str() );
			return false;
		}
	}
}

// Method for sanitizing list of n integers
bool sanitize_int_list(std::string key, std::vector<std::string> input_for_key, std::vector<std::string> input_key_vals){

	// Check to make sure n value has been passed for key
	if ( (int)input_for_key.size() != std::stoi(input_key_vals[0]) ){
		printf("Parse Error: Key (%s) requires an integer list of length (%s)\n", key.c_str(), input_key_vals[0].c_str());
		return false;
	}

	// Check that each item in the list is an integer
	else {

		for ( std::string val : input_for_key){

			// Integer regex
			if ( !std::regex_search( val, std::regex("\\d+") ) ){

				printf("Parse Error: Value (%s) for key (%s) must be an integer (representation)\n",  
					val.c_str(), key.c_str() );
				return false;
			}
		}
		return true;
	}
}

// Exists flag. For passing boolean flags
bool sanitize_exists(std::string key, std::vector<std::string> input_for_key, std::vector<std::string> input_key_vals ){

	// Check if there are no arguments for the input
	if ( input_for_key.size() == 0 ){
		return true;
	}

	// If arguments passed to key then return an error
	else {
		printf("Parse Error: key (%s) takes no arguments\n", key.c_str() );
		return false;
	} 
}

// Tuple flag check input value exists in tuple
bool sanitize_in_tuple(std::string key, std::vector<std::string> input_for_key, std::vector<std::string> input_key_vals ){
	
	// Check to make sure one value has been passed for key
	if ( input_for_key.size() != 1 ){ 
		printf("Parse Error: key (%s) takes one argument\n", key.c_str() );
		return false; 
	}

	// Check to make sure value exists in tuple
	else {
		if ( std::find( input_key_vals.begin(), input_key_vals.end(), input_for_key[0] ) != input_key_vals.end() ){
			return true; 
		}
		else{

			printf("Parse Error: Invalid value (%s) for key (%s)\n", input_for_key[0].c_str(), key.c_str() );
			printf("\tValid values: ");
			for (std::string val : input_key_vals){ printf("\"%s\" ", val.c_str()); }
			printf("\n");
			return false;
		}
	}
}

// Check whether something is a string
bool sanitize_string(std::string key, std::vector<std::string> input_for_key, std::vector<std::string> input_key_vals ){

	if ( input_for_key.size() != 1 ){ 
		printf("Parse Error: key (%s) takes one argument\n", key.c_str() );
		return false;
	}
	else {
		return true;
	}
}

class cl_input_parser {

	public:
		std::vector<std::string> input_raw;

		// Keys and map to function pointers
		std::vector<std::string> input_keys;
		std::map<std::string, function> input_key_rules;
		std::map<std::string, std::vector<std::string>> input_key_vals;

		// Keys passed on the command line (subset of input keys)
		std::vector<std::string> passed_keys;
		std::map<std::string, std::vector<std::string>> passed_key_data;	

		// Constructors
		cl_input_parser(int argc, char** argv);
		~cl_input_parser();

		// Add key rules
		void add_key_rule(std::string, function, std::vector<std::string> );	
		void map_key_rules(void);

		// Extract values from parser object
		bool no_key_passed(void);
		bool is_key_passed(std::string);
		std::vector<std::string> get_key_values(std::string);
		
		// Show input method
		void show_input_raw(void);
		void show_input_data(void);
};

// Constructor: parse raw input
cl_input_parser::cl_input_parser(int argc, char** argv) {
	for( int i = 1; i < argc; i++){
		this->input_raw.push_back( std::string( argv[i] ) );
	}
}

// Destructor
cl_input_parser::~cl_input_parser(void) { }

// Show input method
void cl_input_parser::show_input_raw(void){
	for ( std::string token : this->input_raw ) { 
		std::cout<<token<<"\n"; 
	}
}

// Show input data
void cl_input_parser::show_input_data(void){

	for ( std::string key : this->passed_keys ){
		
		std::string key_data = "\0";
		for ( std::string d : this->passed_key_data[key] ){
			key_data.append(d);
		} 
		printf("%s : %s\n", key.c_str(), key_data.c_str() );
	}
}

// Add key/key_rule
void cl_input_parser::add_key_rule( std::string key, function f, std::vector<std::string> vals = {} ){
	this->input_keys.push_back( std::string(key) );
	this->input_key_rules[key] = f;
	this->input_key_vals[key] = vals;
}


void cl_input_parser::map_key_rules(void){

	// Loop through available input keys to build a list of passed keys
	for ( std::string key : this->input_keys){		

		// Raise parse error if multiple occurances for key
		if (std::count(this->input_raw.begin(), this->input_raw.end(), key ) > 1 ){
			printf("Parse Error: multiple entries for key (%s)\n", key.c_str() );
			exit(1);
		}

		// Push the key onto the list of passed keys
		if( std::find( this->input_raw.begin(), this->input_raw.end(), key ) != this->input_raw.end() ){
			this->passed_keys.push_back(key);
		}
	}

	// For every passed key, we are going to get a string
	// Note the passed_keys list is guaranteed to be unique
	for ( std::string key : this->passed_keys ){

		// Stores the input for a given key
		std::vector<std::string> input_for_key; 
		bool append = false;
		
		// Loop through raw input
		for ( std::string token : this->input_raw ){

			// If we find our key switch on the append and goto next token
			if ( token.compare(key) == 0 ){
				append = true;
				continue;
			}

			// If we come across another token while appending, break the loop
			if ( append && std::find( this->passed_keys.begin(), this->passed_keys.end(), token) != this->passed_keys.end() ){
			 	break;
			}
			else if ( append ){
					input_for_key.push_back(token);
				
			}
		}

		// Call sanitize function on key/input_for_key.
		function key_rule = this->input_key_rules[ key ];

		// If input is valid then push key into map
		if ( key_rule(key, input_for_key, this->input_key_vals[key]) ){
			this->passed_key_data[ key ] = input_for_key;
		}
		else {
			exit(1);
		}
	}
}

// Check if a specific key has been passed to the parser
bool cl_input_parser::is_key_passed(std::string key) {
	return ( std::find( this->passed_keys.begin(), this->passed_keys.end(), key) != this->passed_keys.end() ) ? true : false;
}

// Check if no keys have been passed to the parser
bool cl_input_parser::no_key_passed( void ){
	return ( this->passed_keys.size() == 0 ) ? true : false;
}

// Return vector of tokens associated with a given key
std::vector<std::string> cl_input_parser::get_key_values(std::string key){

	if ( std::find( this->passed_keys.begin(), this->passed_keys.end(), key) != this->passed_keys.end() ){
		return this->passed_key_data[key];
	}
	else {
		printf("Key Error: Key (%s) not found \n", key.c_str());
		exit(1);
	}
}
