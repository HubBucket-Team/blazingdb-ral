/*
 * dataloader.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "DataLoader.h"
#include <arrow/io/interfaces.h>
#include <memory>

namespace ral {
namespace io {


template <typename DataProvider, typename FileParser>
data_loader<DataProvider, FileParser>::data_loader(FileParser parser, DataProvider data_provider) {
	// TODO Auto-generated constructor stub
	this->data_provider = data_provider;
	this->parser = parser;
}

template <typename DataProvider, typename FileParser>
data_loader<DataProvider, FileParser>::~data_loader() {
	// TODO Auto-generated destructor stub
}

template <typename DataProvider, typename FileParser>
void data_loader<DataProvider, FileParser>::load_data(std::vector<gdf_column_cpp> & columns, std::vector<bool> include_column){
	while(this->data_provider.has_next()){
		std::shared_ptr<arrow::io::RandomAccessFile> file = this->data_provider.get_next();
		if(file != nullptr){
			parser.parse(file,columns,include_column);
		}
	}
}

} /* namespace io */
} /* namespace ral */
