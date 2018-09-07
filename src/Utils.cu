#include "Utils.cuh"

void print_column(gdf_column * column){

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
	cudaFree(column->data);
	cudaFree(column->valid);
	//delete column;
	//column = nullptr;
}

void realloc_gdf_column(gdf_column * column, size_t size, size_t width){
	free_gdf_column(column);

	create_gdf_column(column,column->dtype,size,nullptr,width);
}

void create_gdf_column(gdf_column * column, gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value){
	char * data;
	gdf_valid_type * valid_device;

	//so allocations are supposed to be 64byte aligned
	size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64;
	cudaError_t cuda_error = cudaMalloc((void **) &valid_device, allocation_size_valid);

	//assume all relevant bits are set to on
	thrust::constant_iterator<unsigned char> valid(255);
	thrust::device_vector<unsigned char> tester(allocation_size_valid);
	thrust::copy(valid, valid + allocation_size_valid, tester.begin());

	thrust::copy(thrust::cuda::par,valid, valid + allocation_size_valid, thrust::detail::make_normal_iterator(valid_device) );
	cuda_error = cudaMalloc((void **) &data,width_per_value * num_values);

	gdf_error error = gdf_column_view(column,(void *) data, valid_device,num_values,type);
	if(input_data != nullptr){
		cudaMemcpy(data,input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
	}

	column->null_count = 0;
}
