#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>


template <typename HostDataType>
void print_column(gdf_column * column){

	HostDataType * host_data_out = new HostDataType[column->size];
	char * host_valid_out;

	if(column->size % GDF_VALID_BITSIZE != 0){
		host_valid_out = new char[(column->size + (GDF_VALID_BITSIZE - (column->size % GDF_VALID_BITSIZE)))/GDF_VALID_BITSIZE];
	}else{
		host_valid_out = new char[column->size / GDF_VALID_BITSIZE];
	}

	int column_width;
	get_column_byte_width(column, &column_width);

	cudaMemcpy(host_data_out,column->data,column_width * column->size, cudaMemcpyDeviceToHost);
	if (column->valid != nullptr)
		cudaMemcpy(host_valid_out,column->valid,sizeof(gdf_valid_type) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
	else
		std::cout<<"Valid is null\n";

	std::cout<<"Printing Column address ptr: "<<column<<", Size: "<<column->size<<"\n"<<std::flush;

	for(int i = 0; i < column->size; i++){
		int col_position = i / GDF_VALID_BITSIZE;
		int bit_offset = GDF_VALID_BITSIZE - (i % GDF_VALID_BITSIZE);
		std::cout<<"host_data_out["<<i<<"] = "<<(host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
	}

	delete[] host_data_out;
	delete[] host_valid_out;

	std::cout<<std::endl<<std::endl;
}

void free_gdf_column(gdf_column * column);

#endif /* UTILS_CUH_ */
