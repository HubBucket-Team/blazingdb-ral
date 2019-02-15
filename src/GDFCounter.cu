/*
 * GDFCounter.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

 #include "GDFCounter.cuh"
 #include <iostream>
#include "cuDF/Allocator.h"

GDFRefCounter* GDFRefCounter::Instance=0;

void GDFRefCounter::register_column(gdf_column* col_ptr){

    if(col_ptr != nullptr){
        std::lock_guard<std::mutex> lock(gc_mutex);
        gdf_column * map_key = {col_ptr};

        if(map.find(map_key) == map.end()){
            map[map_key]=1;
        }
    }
}

void GDFRefCounter::deregister_column(gdf_column* col_ptr)
{
    if (col_ptr != nullptr) {  // TODO: use exceptions instead jump nulls
        std::lock_guard<std::mutex> lock(gc_mutex);
        gdf_column * map_key = {col_ptr};

        if(map.find(map_key) != map.end()){
            map[map_key]=0; //deregistering
        }
    }
}

void GDFRefCounter::increment(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    gdf_column * map_key = {col_ptr};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]!=0){ //is already deregistered
            map[map_key]++;
        }
    }
}

void GDFRefCounter::free(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    gdf_column * map_key = {col_ptr};

    if(map.find(map_key)!=map.end()){
        map.erase(map_key);

        try {
            cuDF::Allocator::deallocate(map_key->data);
            if (map_key->valid != nullptr) {
                cuDF::Allocator::deallocate(map_key->valid);
            }
            delete map_key;
        }
        catch (const cuDF::Allocator::Exception& exception) {
            std::cerr << exception.what() << std::endl;
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }
}

void GDFRefCounter::decrement(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    gdf_column * map_key = {col_ptr};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]>0){
            map[map_key]--;

            if(map[map_key]==0){
                map.erase(map_key);

                try {
                    cuDF::Allocator::deallocate(map_key->data);
                    if (map_key->valid != nullptr) {
                        cuDF::Allocator::deallocate(map_key->valid);
                    }
                    delete map_key;
                }
                catch (const cuDF::Allocator::Exception& exception) {
                    std::cerr << exception.what() << std::endl;
                    cudaDeviceReset();
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
}

bool GDFRefCounter::contains_column(gdf_column * ptrs){
	if(this->map.find(ptrs) == this->map.end()){
		return false;
	}
	return true;
}

void GDFRefCounter::show_summary()
{
    std::cout<<"--------------------- RefCounter Summary -------------------------\n";

    for (auto& iter : this->map)
        std::cout << "Ptr: " << iter.first << " count: " << iter.second <<"\n";
    
    std::cout<<"Size: "<<get_map_size()<<"\n";

    std::cout<<"------------------ End RefCounter Summary -------------------------\n";
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

std::size_t GDFRefCounter::column_ref_value(gdf_column* column) {
    std::lock_guard<std::mutex> lock(gc_mutex);
    auto it = map.find(column);
    if (it == map.end()) {
        return 0;
    }
    return it->second;
}
