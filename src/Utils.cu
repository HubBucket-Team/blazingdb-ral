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

void update_null_count(gdf_column * column)
{
    int count;
    gdf_error result = gdf_count_nonzero_mask(column->valid, column->size, &count);
    assert(result == GDF_SUCCESS);
    column->null_count = column->size - static_cast<gdf_size_type>(count);
}