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
		void increment(gdf_column* key_ptr); //si no esta registrado, no hace nada  -- register()
		void decrement(gdf_column* key_ptr); //deregister(): 

		/*void register(gdf_column* key_ptr);
		void deregister(gdf_column* key_ptr);*/

		static GDFRefCounter* getInstance();
};

#endif /* GDFCOUNTER_H_ */