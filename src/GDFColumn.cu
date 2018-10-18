/*
 * GDFColumn.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#include "GDFColumn.cuh"

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

gdf_column_cpp::gdf_column_cpp(void* _data, gdf_valid_type* _valid, gdf_dtype _dtype, size_t _size, gdf_size_type _null_count)
{
    column.data = _data;
    column.valid = _valid;
    column.size = _size;
    column.dtype = _dtype;
    column.null_count = _null_count;
    int byte_width;
    get_column_byte_width(this,&byte_width);
    this->allocated_size_data = size * byte_width;
    this->allocated_size_valid = (((((this->size()+ 7 ) / 8) + 63 ) / 64) * 64);

}

gdf_column_cpp::gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    create_gdf_column(type, num_values, input_data, width_per_value);

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
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));

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
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));

}

gdf_column* gdf_column_cpp::get_gdf_column()
{
    return &column;
}

//TODO: needs to be implemented for efficiency though not strictly necessary
void gdf_column_cpp::compact(){
    if( this->allocation_size_valid != (((((this->size()+ 7 ) / 8) + 63 ) / 64) * 64)){
    	//compact valid allcoation

    }

    int byte_width;
    get_column_byte_width(this,&byte_width);
    if(this->allocated_size_data != (this->size() * byte_width)){
    	//compact data allocation
    }
}

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    assert(type != GDF_invalid);

    char * data;
    gdf_valid_type * valid_device;

    this->allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

    cudaMalloc((void **) &valid_device, allocation_size_valid);

    cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid); //assume all relevant bits are set to on

    this->allocated_size_data = width_per_value * num_values;
    cudaMalloc((void **) &data, this->allocated_size_data);

    gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

    if(input_data != nullptr){
        cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
    }


    GDFRefCounter::getInstance()->register_column(&this->column);

}

void gdf_column_cpp::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	create_gdf_column(type, size, nullptr, width);
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
	return GDFRefCounter::getInstance()->contains_column(this->data());
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
