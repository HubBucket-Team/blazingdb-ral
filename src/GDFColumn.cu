/*
 * GDFColumn.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#include <arrow/util/bit-util.h>
//readme:  use bit-utils to compute valid.size in a standard way
// see https://github.com/apache/arrow/blob/e34057c4b4be8c7abf3537dd4998b5b38919ba73/cpp/src/arrow/ipc/writer.cc#L66

#include "GDFColumn.cuh"
#include "gdf_wrapper/gdf_wrapper.cuh"
#include "cuDF/Allocator.h"

gdf_column_cpp::gdf_column_cpp()
{
	column = nullptr;
    this->allocated_size_data = 0;
    this->allocated_size_valid = 0;
    this->is_ipc_column = false;

}
/*
gdf_column_cpp::gdf_column_cpp(void* _data, gdf_valid_type* _valid, gdf_dtype _dtype, size_t _size, gdf_size_type _null_count, const std::string &column_name)
{

    column->data = _data;
    column->valid = _valid;
    column->size = _size;
    column->dtype = _dtype;
    column->null_count = _null_count;
    int byte_width;
    get_column_byte_width(&column,&byte_width);
    this->allocated_size_data = _size * byte_width;
    //Todo: To validate valid_size
    this->allocated_size_valid = gdf_get_num_chars_bitmask(_size);
    this->column_name = column_name;

}*/

gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& col)
{
    column = col.column;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->column_name= col.column_name;
    this->is_ipc_column = col.is_ipc_column;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(col.column));

}

gdf_column_cpp::gdf_column_cpp(gdf_column_cpp& col)
{
	column = col.column;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->column_name= col.column_name;
    this->is_ipc_column = col.is_ipc_column;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(col.column));

}


void gdf_column_cpp::set_name(std::string name){
		this->column_name = name;
		//why the fuck isnt the name a fixed size

		this->column->col_name = new char[name.size() + 1];
		this->column->col_name[name.size()] = 0;
		std::copy(name.begin(), name.end(),
				this->column->col_name);



	}
void gdf_column_cpp::set_name_cpp_only(std::string name){
	this->column_name = name;
}

void gdf_column_cpp::delete_set_name(std::string name){
	delete [] this->column->col_name;
	this->set_name(name);
}
std::string gdf_column_cpp::name() const{
	return this->column_name;
}
gdf_column_cpp gdf_column_cpp::clone(std::string name)  // TODO clone needs to register
{




	char * data_dev = nullptr;
	char * valid_dev = nullptr;

    try {
        cuDF::Allocator::allocate((void**)&data_dev, allocated_size_data);
        if (column->valid != nullptr) {
            cuDF::Allocator::allocate((void**)&valid_dev, allocated_size_valid);
        }
    }
    catch (const cuDF::Allocator::Exception& exception) {
        std::cerr << exception.what() << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    CheckCudaErrors(cudaMemcpy(data_dev, this->column->data, this->allocated_size_data, cudaMemcpyDeviceToDevice));
	// print_gdf_column(this->get_gdf_column());
    if (this->column->valid != nullptr) {
	    CheckCudaErrors(cudaMemcpy(valid_dev, this->column->valid, this->allocated_size_valid, cudaMemcpyDeviceToDevice));
    }


	gdf_column_cpp col1;
    col1.column = new gdf_column;
	*col1.column = *this->column;
	col1.column->data = (void *) data_dev;
	col1.column->valid =(gdf_valid_type *) valid_dev;
	col1.allocated_size_data = this->allocated_size_data;
	col1.allocated_size_valid = this->allocated_size_valid;
	col1.is_ipc_column = false;
	if(name == ""){
		col1.set_name(this->column_name);
	}else{
		col1.set_name(name);
	}



//	print_gdf_column(col1.get_gdf_column());

	GDFRefCounter::getInstance()->register_column(col1.column);

	return col1;
}

void gdf_column_cpp::operator=(const gdf_column_cpp& col)
{
	column = col.column;
    this->allocated_size_data = col.allocated_size_data;
    this->allocated_size_valid = col.allocated_size_valid;
    this->set_name(col.column_name);
    this->is_ipc_column = col.is_ipc_column;
    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(col.column));

}

gdf_column* gdf_column_cpp::get_gdf_column()
{
    return column;
}

void gdf_column_cpp::resize(size_t new_size){
	this->column->size = new_size;
}
//TODO: needs to be implemented for efficiency though not strictly necessary
gdf_error gdf_column_cpp::compact(){
    if( this->allocated_size_valid != arrow::BitUtil::PaddedLength(arrow::BitUtil::BytesForBits(this->size()))){
    	//compact valid allcoation

    }

    int byte_width;
    get_column_byte_width(this->get_gdf_column(),&byte_width);
    if(this->allocated_size_data != (this->size() * byte_width)){
    	//compact data allocation
    }
    return GDF_SUCCESS;
}

void gdf_column_cpp::update_null_count()
{
    int count;
    gdf_error result = gdf_count_nonzero_mask(this->column->valid, this->column->size, &count);
    assert(result == GDF_SUCCESS);
    this->column->null_count = this->column->size - static_cast<gdf_size_type>(count);
}

void gdf_column_cpp::allocate_set_valid(){
	this->column->valid = allocate_valid();
}
gdf_valid_type * gdf_column_cpp::allocate_valid(){
	size_t num_values = this->size();
    gdf_valid_type * valid_device;
	this->allocated_size_valid = arrow::BitUtil::PaddedLength(arrow::BitUtil::BytesForBits(num_values)); //so allocations are supposed to be 64byte aligned

    try {
        cuDF::Allocator::allocate((void**)&valid_device, allocated_size_valid);
    }
    catch (const cuDF::Allocator::Exception& exception) {
        std::cerr << exception.what() << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    CheckCudaErrors(cudaMemset(valid_device, (gdf_valid_type)255, allocated_size_valid)); //assume all relevant bits are set to on
	return valid_device;
}

void gdf_column_cpp::create_gdf_column_for_ipc(gdf_dtype type, void * col_data,gdf_valid_type * valid_data,size_t num_values,std::string column_name){
    assert(type != GDF_invalid);
    int width;

    //TODO crate column here
    this->column = new gdf_column;
    gdf_column_view(this->column, col_data, valid_data, num_values, type);
    get_column_byte_width(this->column, &width);
    this->allocated_size_data = num_values * width;
    this->allocate_valid();
    is_ipc_column = true;
    this->set_name(column_name);
}

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value, const std::string &column_name)
{
    assert(type != GDF_invalid);
    this->column = new gdf_column;

    //TODO: this is kind of bad its a chicken and egg situation with column_view requiring a pointer to device and allocate_valid
    //needing to not require numvalues so it can be called rom outside
    this->get_gdf_column()->size = num_values;
    char * data;
    is_ipc_column = false;

    gdf_valid_type * valid_device = allocate_valid();
    this->allocated_size_data = (((width_per_value * num_values) + 63) /64) * 64;

    try {
        cuDF::Allocator::allocate((void**)&data, allocated_size_data);
    }
    catch (const cuDF::Allocator::Exception& exception) {
        std::cerr << exception.what() << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    gdf_column_view(this->column, (void *) data, valid_device, num_values, type);
    this->set_name(column_name);
    if(input_data != nullptr){
        CheckCudaErrors(cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice));
    }


    GDFRefCounter::getInstance()->register_column(this->column);

}
void gdf_column_cpp::create_gdf_column(gdf_column * column){
	this->column = column;
	int width_per_value;
	gdf_error err = get_column_byte_width(column, &width_per_value);

	//TODO: we are assuming they are not padding,
	this->allocated_size_data = width_per_value * column->size;
	if(column->valid != nullptr){
        this->allocated_size_valid = arrow::BitUtil::PaddedLength(arrow::BitUtil::BytesForBits(column->size)); //so allocations are supposed to be 64byte aligned
	}
	this->is_ipc_column = false;
    if (column->col_name)
    	this->column_name = std::string(column->col_name);

    GDFRefCounter::getInstance()->register_column(this->column);
}
/*
void gdf_column_cpp::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
	const std::string col_name = this->column_name;
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	this->create_gdf_column(type, size, nullptr, width, col_name);
}*/

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
	if(this->is_ipc_column){
		//TODO: this is a big memory leak. we probably just need to have anothe reference
		//counter, the valid pointer was allocated on our side
		//we cant free it here because we dont know if this ipc column is used somewhere else
	}else{
	    GDFRefCounter::getInstance()->decrement(this->column);
	}

}
bool gdf_column_cpp::is_ipc(){
	return this->is_ipc_column;
}
void* gdf_column_cpp::data(){
    return column->data;
}

gdf_valid_type* gdf_column_cpp::valid(){
    return column->valid;
}
gdf_size_type gdf_column_cpp::size(){
    return column->size;
}

gdf_dtype gdf_column_cpp::dtype(){
    return column->dtype;
}

gdf_size_type gdf_column_cpp::null_count(){
    return column->null_count;
}

gdf_dtype_extra_info gdf_column_cpp::dtype_info(){
    return column->dtype_info;
}

void gdf_column_cpp::set_dtype(gdf_dtype dtype){
    column->dtype=dtype;
}

std::size_t gdf_column_cpp::get_valid_size() const {
    return allocated_size_valid;
}
