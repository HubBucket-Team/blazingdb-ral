/*
 * GDFColumn.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#include "GDFColumn.cuh"

#include <gdf/utils.h>

gdf_column_cpp::gdf_column_cpp()
{
    column.data = nullptr;
    column.valid = nullptr;
    column.size = 0;
    column.dtype = GDF_invalid;
    column.null_count = 0;
    this->allocated_size_data = 0;
    this->allocated_size_valid = 0;

}

gdf_column_cpp::gdf_column_cpp(void* _data, gdf_valid_type* _valid, gdf_dtype _dtype, size_t _size, gdf_size_type _null_count, const std::string &column_name)
{
    column.data = _data;
    column.valid = _valid;
    column.size = _size;
    column.dtype = _dtype;
    column.null_count = _null_count;
    int byte_width;
    get_column_byte_width(&column,&byte_width);
    this->allocated_size_data = _size * byte_width;
    //Todo: To validate valid_size
    this->allocated_size_valid = gdf_get_num_chars_bitmask(_size);
    this->column_name = column_name;

}

gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->column_name= col.column_name;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));

}

gdf_column_cpp::gdf_column_cpp(gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->column_name= col.column_name;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));

}

gdf_column_cpp gdf_column_cpp::clone()  // TODO clone needs to register
{
	void* data_dev = nullptr;
	void* valid_dev = nullptr;

	CheckCudaErrors(cudaMalloc(&data_dev, this->allocated_size_data));
	CheckCudaErrors(cudaMemcpy(data_dev, this->column.data, this->allocated_size_data, cudaMemcpyDeviceToDevice));

	print_gdf_column(this->get_gdf_column());
    if (this->column.valid != nullptr) {
    	CheckCudaErrors(cudaMalloc(&valid_dev, this->allocated_size_valid));
	    CheckCudaErrors(cudaMemcpy(valid_dev, this->column.valid, this->allocated_size_valid, cudaMemcpyDeviceToDevice));
    }

	gdf_column_cpp col1(data_dev,
						(gdf_valid_type*) valid_dev,
						this->column.dtype,
						this->column.size,
						this->column.null_count,
						this->column_name);

//	print_gdf_column(col1.get_gdf_column());

	GDFRefCounter::getInstance()->register_column(&col1.column);

	return col1;
}

void gdf_column_cpp::operator=(const gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->column_name= col.column_name;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));

}

gdf_column* gdf_column_cpp::get_gdf_column()
{
    return &column;
}

//TODO: needs to be implemented for efficiency though not strictly necessary
gdf_error gdf_column_cpp::compact(){
    if( this->allocated_size_valid != (((((this->size()+ 7 ) / 8) + 63 ) / 64) * 64)){
    	//compact valid allcoation

    }

    int byte_width;
    get_column_byte_width(this->get_gdf_column(),&byte_width);
    if(this->allocated_size_data != (this->size() * byte_width)){
    	//compact data allocation
    }
}

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value, const std::string &column_name)
{
    assert(type != GDF_invalid);

    this->column_name = column_name;

    char * data;
    gdf_valid_type * valid_device;

    this->allocated_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

    CheckCudaErrors(cudaMalloc(&valid_device, allocated_size_valid));

    CheckCudaErrors(cudaMemset(valid_device, (gdf_valid_type)255, allocated_size_valid)); //assume all relevant bits are set to on

    this->allocated_size_data = (((width_per_value * num_values) + 63) /64) * 64;
    CheckCudaErrors(cudaMalloc(&data, this->allocated_size_data));

    gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

    if(input_data != nullptr){
        CheckCudaErrors(cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice));
    }


    GDFRefCounter::getInstance()->register_column(&this->column);

}

void gdf_column_cpp::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
	const std::string col_name = this->column_name;
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	this->create_gdf_column(type, size, nullptr, width, col_name);
}

gdf_error gdf_column_cpp::gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype)
{
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    column->null_count = 0;
    return GDF_SUCCESS;
}

gdf_column_cpp::~gdf_column_cpp()
{
    GDFRefCounter::getInstance()->decrement(&this->column);
}
bool gdf_column_cpp::is_ipc(){
	return !GDFRefCounter::getInstance()->contains_column(std::make_pair(this->data(),this->valid()));
}
void* gdf_column_cpp::data(){
    return column.data;
}

gdf_valid_type* gdf_column_cpp::valid(){
    return column.valid;
}
gdf_size_type gdf_column_cpp::size(){
    return column.size;
}

gdf_dtype gdf_column_cpp::dtype(){
    return column.dtype;
}

gdf_size_type gdf_column_cpp::null_count(){
    return column.null_count;
}

gdf_dtype_extra_info gdf_column_cpp::dtype_info(){
    return column.dtype_info;
}

void gdf_column_cpp::set_dtype(gdf_dtype dtype){
    column.dtype=dtype;
}
