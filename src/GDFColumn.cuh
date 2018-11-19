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
#include "Utils.cuh"
#include <string>

class gdf_column_cpp
{
	private:
		gdf_column * column;
		size_t allocated_size_data;
		size_t allocated_size_valid;
		std::string column_name;
		gdf_valid_type * allocate_valid();
		void set_name(std::string name);
		int8_t is_ipc_column;
	//	gdf_column_cpp(void* _data, gdf_valid_type* _valid, gdf_dtype _dtype, size_t _size, gdf_size_type _null_count, const std::string &column_name = "");
	public:

		//TODO: theres a memory leak here, if we change the name
		void delete_set_name(std::string name);
    void* data();
    void allocate_set_valid();
    gdf_valid_type* valid();
    std::string name() const;

    gdf_size_type size();

    gdf_dtype dtype();

    gdf_size_type null_count();

    gdf_dtype_extra_info dtype_info();

	void set_dtype(gdf_dtype dtype);
	bool is_ipc();

	gdf_column_cpp();

	void create_gdf_column_for_ipc(gdf_dtype type, void * col_data,gdf_valid_type * valid_data,size_t size,std::string column_name);

	gdf_column_cpp(const gdf_column_cpp& col);

	gdf_column_cpp(gdf_column_cpp& col);

	void operator=(const gdf_column_cpp& col);

	gdf_column_cpp clone(std::string name = "");

	gdf_column* get_gdf_column();

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value, const std::string &column_name = "");

	//void realloc_gdf_column(gdf_dtype type, size_t size, size_t width);

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

	~gdf_column_cpp();

	void resize(size_t new_size);
	gdf_error compact();

public:
    std::size_t get_valid_size() const;
};

#endif /* GDFCOLUMN_H_ */
