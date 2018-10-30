/*
 * GDFCounter.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

 #include "GDFCounter.cuh"
 #include <iostream>

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
