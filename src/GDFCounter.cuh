/*
 * GDFCounter.h
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#ifndef GDFCOUNTER_H_
#define GDFCOUNTER_H_

#include <gdf/gdf.h>
#include <map>
#include <mutex>

class GDFRefCounter
{
	private:
		GDFRefCounter();

		static GDFRefCounter* Instance;
		std::mutex gc_mutex;

		std::map<gdf_column*, size_t> map; // std::map<key_ptr, ref_counter>
	
	public:
		void increment(gdf_column* key_ptr);
		void decrement(gdf_column* key_ptr);

		void register_column(gdf_column* key_ptr);
		void deregister_column(gdf_column* key_ptr);
		void swap_key(gdf_column* key_ptr1, gdf_column* key_ptr2);

		void free_if_deregistered(gdf_column* key_ptr);

		size_t get_map_size();

		static GDFRefCounter* getInstance();
};

#endif /* GDFCOUNTER_H_ */