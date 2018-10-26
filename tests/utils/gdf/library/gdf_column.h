#pragma once 

#include <map>
#include <mutex>

#include "gdf/gdf.h"

namespace gdf {
namespace library {

constexpr size_t GDF_VALID_BITSIZE{(sizeof(gdf_valid_type) * 8)};

inline gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size) { return (( size + ( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE ); }

inline gdf_error get_column_byte_width(gdf_column * col, int * width){
	
	switch(col->dtype) {

	case GDF_INT8 :
		*width = 1;
		break;
	case GDF_INT16 :
		*width = 2;
		break;
	case GDF_INT32 :
		*width = 4;
		break;
	case GDF_INT64 :
		*width = 8;
		break;
	case GDF_FLOAT32 :
		*width = 4;
		break;
	case GDF_FLOAT64 :
		*width = 8;
		break;
	case GDF_DATE32 :
		*width = 4;
		break;
	case GDF_DATE64 :
		*width = 8;
		break;
	case GDF_TIMESTAMP :
		*width = 8;
		break;
	case GDF_UINT8 :
		*width = 1;
		break;
	case GDF_UINT16:
    	*width = 2;
		break;
	case GDF_UINT32:
    	*width = 4;
		break;
	case GDF_UINT64:
		*width = 8;
		break;
	default :
		*width = -1;
		return GDF_UNSUPPORTED_DTYPE;
	}

	return GDF_SUCCESS;
}


typedef std::pair<void*, gdf_valid_type*> rc_key_t; // std::pair<void* data, gdf_valid_type* valid>

class GDFRefCounter
{
	private:
		GDFRefCounter();

		static GDFRefCounter* Instance;

		std::mutex gc_mutex;

		std::map<rc_key_t, size_t> map; // std::map<key_ptr, ref_counter>

	public:
		void increment(gdf_column* col_ptr);

		void decrement(gdf_column* col_ptr);

		void register_column(gdf_column* col_ptr);

		void deregister_column(gdf_column* col_ptr);

		void free_if_deregistered(gdf_column* col_ptr);

		size_t get_map_size();

		bool contains_column(rc_key_t ptrs);
		static GDFRefCounter* getInstance();
};



GDFRefCounter* GDFRefCounter::Instance=0;

void GDFRefCounter::register_column(gdf_column* col_ptr){

    if(col_ptr != nullptr){
        std::lock_guard<std::mutex> lock(gc_mutex);
        rc_key_t map_key = {col_ptr->data, col_ptr->valid};
        
        if(map.find(map_key) == map.end()){
            map[map_key]=1;
        }
    }
}

void GDFRefCounter::deregister_column(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key) != map.end()){
        map[map_key]=0; //deregistering
    }
}

void GDFRefCounter::free_if_deregistered(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]==0){
            map.erase(map_key);
            //@todo: memory leak
            // cudaFree(map_key.first); //data
            // cudaFree(map_key.second); //valid
        }
    }
}

void GDFRefCounter::increment(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]!=0){ //is already deregistered
            map[map_key]++;
        }
    }
}

void GDFRefCounter::decrement(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]>0){
            map[map_key]--;

            if(map[map_key]==0){
                map.erase(map_key);
                //@todo: memory leak
                // cudaFree(map_key.first); //data
                // cudaFree(map_key.second); //valid
            }
        }
    }
}

bool GDFRefCounter::contains_column(rc_key_t ptrs){
	if(this->map.find(ptrs) == this->map.end()){
		return false;
	}
	return true;
}

GDFRefCounter::GDFRefCounter()
{

}

// Testing purposes
size_t GDFRefCounter::get_map_size()
{
    return map.size();
}

GDFRefCounter* GDFRefCounter::getInstance()
{
    if(!Instance)
        Instance=new GDFRefCounter();
    return Instance;
}



class gdf_column_cpp
{
	private:
		gdf_column column;
		size_t allocated_size_data;
		size_t allocated_size_valid;


	public:

    void* data();

    gdf_valid_type* valid();

    gdf_size_type size();

    gdf_dtype dtype();

    gdf_size_type null_count();

    gdf_dtype_extra_info dtype_info();

	void set_dtype(gdf_dtype dtype);
	bool is_ipc();

	gdf_column_cpp();

	gdf_column_cpp(void* _data, gdf_valid_type* _valid, gdf_dtype _dtype, size_t _size, gdf_size_type _null_count);

	gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	gdf_column_cpp(const gdf_column_cpp& col);

	gdf_column_cpp(gdf_column_cpp& col);

	void operator=(const gdf_column_cpp& col);

	gdf_column_cpp clone();

	gdf_column* get_gdf_column();

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	void realloc_gdf_column(gdf_dtype type, size_t size, size_t width);

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

	~gdf_column_cpp();

	gdf_error compact();
};


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
    get_column_byte_width(&column,&byte_width);
    this->allocated_size_data = _size * byte_width;
    //Todo: To validate valid_size
    this->allocated_size_valid = gdf_get_num_chars_bitmask(_size);

}

gdf_column_cpp::gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    this->create_gdf_column(type, num_values, input_data, width_per_value);

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

gdf_column_cpp gdf_column_cpp::clone()
{
	void* data_dev = nullptr;
	void* valid_dev = nullptr;

	(cudaMalloc(&data_dev, this->allocated_size_data));
	(cudaMemcpy(data_dev, this->column.data, this->allocated_size_data, cudaMemcpyDeviceToDevice));

    if (this->column.valid != nullptr) {
    	(cudaMalloc(&valid_dev, this->allocated_size_valid));
	    (cudaMemcpy(valid_dev, this->column.valid, this->allocated_size_valid, cudaMemcpyDeviceToDevice));
    }

	gdf_column_cpp col1(data_dev,
						(gdf_valid_type*) valid_dev,
						this->column.dtype,
						this->column.size,
						this->column.null_count);

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

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    assert(type != GDF_invalid);

    char * data;
    gdf_valid_type * valid_device;

    this->allocated_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

    cudaMalloc((void **) &valid_device, allocated_size_valid);

    cudaMemset(valid_device, (gdf_valid_type)255, allocated_size_valid); //assume all relevant bits are set to on

    this->allocated_size_data = ((width_per_value * num_values) + 63 /64) * 64;
    cudaMalloc((void **) &data, this->allocated_size_data);

    gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

    if(input_data != nullptr){
        cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
    }


    GDFRefCounter::getInstance()->register_column(&this->column);

}

void gdf_column_cpp::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	this->create_gdf_column(type, size, nullptr, width);
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
	return GDFRefCounter::getInstance()->contains_column(std::make_pair(this->data(),this->valid()));
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

using GdfColumn = gdf_column_cpp;

}//container
}//gdf
