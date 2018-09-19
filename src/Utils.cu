#include "Utils.cuh"

void print_column(gdf_column * column){

	//std::cout<<column->size<<"!"<<std::endl;
	char * host_data_out = new char[column->size];
	char * host_valid_out;

	if(column->size % 8 != 0){
		host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
	}else{
		host_valid_out = new char[column->size / 8];
	}

	cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);

	std::cout<<"Printing Column"<<std::endl;

	for(int i = 0; i < column->size; i++){
		int col_position = i / 8;
		int bit_offset = 8 - (i % 8);
		std::cout<<"host_data_out["<<i<<"] = "<<((int)host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
	}

	delete[] host_data_out;
	delete[] host_valid_out;

	std::cout<<std::endl<<std::endl;
}

void free_gdf_column(gdf_column * column){
	//cudaFree(column->data);
	//cudaFree(column->valid);
	//delete column;
	//column = nullptr;
}

void realloc_gdf_column(gdf_column * column, size_t size, size_t width){
	//free_gdf_column(column);

	create_gdf_column(column,column->dtype,size,nullptr,width);
}

void create_gdf_column(gdf_column * column, gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value){
	char * data;
	gdf_valid_type * valid_device;

	//so allocations are supposed to be 64byte aligned
	size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64;

	cudaError_t cuda_error = cudaMalloc((void **) &valid_device, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	//assume all relevant bits are set to on
	cuda_error = cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	cuda_error = cudaMalloc((void **) &data, width_per_value * num_values);
	assert(cuda_error == cudaSuccess);

	gdf_error error = gdf_column_view(column, (void *) data, valid_device, num_values, type);
	assert(error == GDF_SUCCESS);

	if(input_data != nullptr){
		cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
	}

	column->null_count = 0;
}

gdf_col_pointer create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value){
	char * data;
	gdf_valid_type * valid_device;

	//so allocations are supposed to be 64byte aligned
	size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64;

	cudaError_t cuda_error = cudaMalloc((void **) &valid_device, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	//assume all relevant bits are set to on
	cuda_error = cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	cuda_error = cudaMalloc((void **) &data, width_per_value * num_values);
	assert(cuda_error == cudaSuccess);

	// Create a new instance of a gdf_column with a custom deleter that will free
    // the associated device memory when it eventually goes out of scope
    auto deleter = [](gdf_column* col){col->size = 0; cudaFree(col->data); cudaFree(col->valid); };
    gdf_col_pointer the_column{new gdf_column, deleter};

	the_column->data = data;
	the_column->valid = valid_device;
	the_column->size = num_values;
    the_column->dtype = type;
	the_column->null_count = 0;

	/*gdf_error error = gdf_column_view(column, (void *) data, valid_device, num_values, type);
	assert(error == GDF_SUCCESS);*/

	if(input_data != nullptr){
		cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
	}

	return the_column;
}

gdf_col_shared_pointer create_gdf_shared_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value){
	char * data;
	gdf_valid_type * valid_device;

	//so allocations are supposed to be 64byte aligned
	size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64;

	cudaError_t cuda_error = cudaMalloc((void **) &valid_device, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	//assume all relevant bits are set to on
	cuda_error = cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid);
	assert(cuda_error == cudaSuccess);

	cuda_error = cudaMalloc((void **) &data, width_per_value * num_values);
	assert(cuda_error == cudaSuccess);

	// Create a new instance of a gdf_column with a custom deleter that will free
    // the associated device memory when it eventually goes out of scope
    auto deleter = [](gdf_column* col){col->size = 0; cudaFree(col->data); cudaFree(col->valid); };
    gdf_col_shared_pointer the_column{new gdf_column, deleter};

	the_column->data = data;
	the_column->valid = valid_device;
	the_column->size = num_values;
    the_column->dtype = type;
	the_column->null_count = 0;

	/*gdf_error error = gdf_column_view(column, (void *) data, valid_device, num_values, type);
	assert(error == GDF_SUCCESS);*/

	if(input_data != nullptr){
		cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
	}

	return the_column;
}

/*gdf_column* gdf_column_cpp::get_gdf_column()
{
	return &column;
}

gdf_column_cpp::gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
	create_gdf_column(type, num_values, input_data, width_per_value);
}

gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& col)
{
	this->column.data = col.column.data;
	this->column.valid = col.column.valid;
	this->column.size = col.column.size;
	this->column.dtype = col.column.dtype;
	this->column.null_count = col.column.null_count;
}

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
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
	cudaFree(this->column.data);
	cudaFree(this->column.valid);
}*/
