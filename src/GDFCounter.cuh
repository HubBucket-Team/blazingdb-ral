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

typedef std::pair<void*, gdf_valid_type*> rc_key_t; // std::pair<void* data, gdf_valid_type* valid>

class GDFRefCounter
{
	private:
		GDFRefCounter();

		static GDFRefCounter* Instance;
		std::mutex gc_mutex;

		std::map<rc_key_t, size_t> map; // std::map<key_ptr, ref_counter>
	
	public:
		void increment(gdf_column* col_ptr);
		void decrement(gdf_column* col_ptr);

		void register_column(gdf_column* col_ptr);
		void deregister_column(gdf_column* col_ptr);

		void free_if_deregistered(gdf_column* col_ptr);

		size_t get_map_size();

		static GDFRefCounter* getInstance();
};

#endif /* GDFCOUNTER_H_ */