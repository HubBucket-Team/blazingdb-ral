#include "Utils.cuh"

void free_gdf_column(gdf_column * column){
	std::cout<<"Deallocating "<<column->data<<std::flush<<"\n";
	cudaFree(column->data);
	cudaFree(column->valid);
	//delete column;
	//column = nullptr;
}
