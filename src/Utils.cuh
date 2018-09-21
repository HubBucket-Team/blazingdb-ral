#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

void print_column(gdf_column * column);

void free_gdf_column(gdf_column * column);

void create_gdf_column(gdf_column * column, gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

/*void realloc_gdf_column(gdf_column * column, size_t size, size_t width);*/

#endif /* UTILS_CUH_ */
