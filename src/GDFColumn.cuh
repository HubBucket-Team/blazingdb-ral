/*
 * GDFColumn.h
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#ifndef GDFCOLUMN_H_
#define GDFCOLUMN_H_

#include <gdf/gdf.h>
#include "GDFCounter.cuh"

class gdf_column_cpp
{
	private:
		gdf_column column;
		//GDFRefCounter* gc;

	public:

    void* data();
    gdf_valid_type* valid();
    gdf_size_type size();
    gdf_dtype dtype();
    gdf_size_type null_count();
    gdf_dtype_extra_info dtype_info();
	void set_dtype(gdf_dtype dtype);

	gdf_column_cpp();

	gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	gdf_column_cpp(const gdf_column_cpp& col);

	gdf_column* get_gdf_column();

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

	~gdf_column_cpp();
};

#endif /* GDFCOLUMN_H_ */