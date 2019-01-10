#include "Utils.cuh"
#include "cuDF/Allocator.h"

void free_gdf_column(gdf_column * column){
	std::cout<<"Deallocating "<<column->data<<std::flush<<"\n";

    try {
        cuDF::Allocator::deallocate(column->data);
        cuDF::Allocator::deallocate(column->valid);
    }
    catch (const cuDF::Allocator::Exception& exception) {
        std::cerr << exception.what() << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
	//delete column;
	//column = nullptr;
}

void gdf_sequence(int32_t* data, size_t size, int32_t init_val){
  auto d_ptr = thrust::device_pointer_cast(data);
  thrust::sequence(d_ptr, d_ptr + size, init_val);
}
