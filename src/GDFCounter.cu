/*
 * GDFCounter.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */
 
 #include "GDFCounter.cuh"
 #include <iostream>

GDFRefCounter* GDFRefCounter::Instance=0;

void GDFRefCounter::register_column(gdf_column* key_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    if(map.find(key_ptr)==map.end())
        map[key_ptr]=1;
}

void GDFRefCounter::deregister_column(gdf_column* key_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    if(map.find(key_ptr)!=map.end())
    {
        map[key_ptr]=0; //deregistering
        //map.erase(key_ptr);
    }
}

void GDFRefCounter::swap_key(gdf_column* key_ptr1, gdf_column* key_ptr2)
{
    auto it = map.find(key_ptr1);
    if (it != map.end())
    {
      const size_t value = it->second;
      map.erase(it);
      map[key_ptr2] = value;
    }
}

void GDFRefCounter::free_if_deregistered(gdf_column* key_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    if(map.find(key_ptr)!=map.end())
    {
        if(map[key_ptr]==0)
        {
            map.erase(key_ptr);
            cudaFree(key_ptr->data);
            cudaFree(key_ptr->valid);
        }
    }
}

void GDFRefCounter::increment(gdf_column* key_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    if(map.find(key_ptr)!=map.end())
        map[key_ptr]++;
}

void GDFRefCounter::decrement(gdf_column* key_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    if(map.find(key_ptr)!=map.end())
    {
        if(map[key_ptr]>0)
        {
            map[key_ptr]--;

            if(map[key_ptr]==0)
            {
                map.erase(key_ptr);
                cudaFree(key_ptr->data);
                cudaFree(key_ptr->valid);
            }
        }
    }
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
