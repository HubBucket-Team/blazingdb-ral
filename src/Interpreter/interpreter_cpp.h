/*
 * interpreter_cpp.h
 *
 *  Created on: Jan 12, 2019
 *      Author: felipe
 */

#ifndef INTERPRETER_CPP_H_
#define INTERPRETER_CPP_H_


#include <cudf.h>
//We have templated cude that has to be in a
//.cuh but we need to be able to include this in cpp code that is not compiled with nvcc
//this wraps that
typedef short column_index_type;
static const short SCALAR_INDEX;
static const short SCALAR_NULL_INDEX;


gdf_error perform_operation();



#endif /* INTERPRETER_CPP_H_ */
