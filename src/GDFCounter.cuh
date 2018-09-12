/*
 * GDFCounter.h
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#ifndef GDFCOUNTER_H_
#define GDFCOUNTER_H_

#include <gdf/gdf.h>
#include <map>
#include <mutex>

typedef gdf_column* key_type;

class gdf_column_cpp
{
	private:
		gdf_column column;

	public:
	gdf_column* get_gdf_column()
	{
		return &column;
	}

	gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
	{
		create_gdf_column(type, num_values, input_data, width_per_value);
	}

	gdf_column_cpp(const gdf_column_cpp& col)
	{
		column->data = col.data;
		column->valid = col.valid;
		column->size = col.size;
		column->dtype = col.dtype;
		column->null_count = 0;

		gc->increment(&col->column);
	}

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
	{
		char * data;
		gdf_valid_type * valid_device;

		size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

		cudaMalloc((void **) &valid_device, allocation_size_valid);	
	
		cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid); //assume all relevant bits are set to on
	
		cudaMalloc((void **) &data, width_per_value * num_values);

		gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

		if(input_data != nullptr){
			cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
		}

		gc->increment(&this->column);
	}

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype)
	{
		column->data = data;
		column->valid = valid;
		column->size = size;
		column->dtype = dtype;
		column->null_count = 0;
		return GDF_SUCCESS;
	}

	~gdf_column_cpp()
	{
		std::lock_guard<std::mutex> lock(gc_mutex);
		gc->decrement(column->data);
	}
};

class GDFRefCounter
{
	std::map<key_type, size_t> map; // std::map<key_ptr, ref_counter>
	void increment(key_type key_ptr);
	void decrement(key_type key_ptr);

	std::mutex gc_mutex;
};

#endif /* GDFCOUNTER_H_ */