/*
 * GDFCounter.cu
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */
 
 #include "GDFCounter.cuh"

 void GDFRefCounter::increment(key_type key_ptr)
 {
     std::lock_guard<std::mutex> lock(gc_mutex);
 
     if(map.find(key_ptr)==map.end())
         map[key_ptr]=1;
     else
         map[key_ptr]++;
 }
 
 void GDFRefCounter::decrement(key_type key_ptr)
 {
     std::lock_guard<std::mutex> lock(gc_mutex);
     map[key_ptr]--;
 
     if(map[key_ptr]==0)
     {
         map.erase(key_ptr);
         cudaFree(key_ptr->data);
         cudaFree(key_ptr->valid);
         free(key_ptr);
     }
 }
