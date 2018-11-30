/*
 * uridataprovider.cpp
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#include "UriDataProvider.h"
#include "Config/BlazingContext.h"
#include "arrow/status.h"
#include "ExceptionHandling/BlazingException.h"
#include <iostream>

namespace ral {
namespace io {

uri_data_provider::uri_data_provider(std::vector<Uri> uris):
		file_uris(uris), opened_files({}), current_file(0), errors({}) {
	// thanks to c++11 we no longer have anything interesting to do here :)

}

uri_data_provider::~uri_data_provider() {
	//TODO: when a shared_ptr to a randomaccessfile goes out of scope does it close files automatically?
	//in case it doesnt we can close that here
	for(size_t file_index = 0; file_index < this->opened_files.size(); file_index++){
		//TODO: perhaps consider capturing status here and complainig if it fails
		this->opened_files[file_index]->Close();
		//
	}
}

std::string uri_data_provider::get_current_user_readable_file_handle(){
	return this->file_uris[this->current_file].toString();
}

bool uri_data_provider::has_next(){
	return this->current_file < (this->opened_files.size() - 1);
}

std::shared_ptr<arrow::io::RandomAccessFile> uri_data_provider::get_next(){
	//TODO: rethrow exceptions so we can do something nicer and just write out to the console :)
	try{
		std::shared_ptr<arrow::io::RandomAccessFile> file =
				BlazingContext::getInstance()->getFileSystemManager()->openReadable(
						this->file_uris[this->current_file]);
		this->current_file++;
		this->opened_files.push_back(file);
		return file;

	}catch(const BlazingInvalidPathException & e){
		std::cout<<e.what()<<std::endl;
		this->errors.push_back(e.what());
		this->current_file++;
		return nullptr;
	}catch(const BlazingFileSystemException & e){
		std::cout<<e.what()<<std::endl;
		this->errors.push_back(e.what());
		this->current_file++;
		return nullptr;	}
}

std::vector<std::string> uri_data_provider::get_errors(){
	return this->errors;
}

} /* namespace io */
} /* namespace ral */
