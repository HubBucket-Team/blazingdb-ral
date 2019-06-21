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

uri_data_provider::uri_data_provider(std::vector<Uri> uris)
	: data_provider(),  file_uris(uris), opened_files({}), current_file(0), errors({}), directory_uris({}), directory_current_file(0) {
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
	if(directory_uris.size() == 0 ){
		return this->file_uris[this->current_file].toString();
	}else{
		return this->directory_uris[this->directory_current_file].toString();
	}

}

bool uri_data_provider::has_next(){
	return this->current_file < this->file_uris.size();
}

void uri_data_provider::reset() {
	this->current_file = 0;
	this->directory_current_file = 0;
}

std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > uri_data_provider::get_all(){
	std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > file_handles;
	while(this->has_next()){
		file_handles.push_back(this->get_next());
	}

	return file_handles;
}


std::shared_ptr<arrow::io::RandomAccessFile> uri_data_provider::get_next(){
	// TODO: Take a look at this later, just calling this function to ensure
	// the uri is in a valid state otherwise throw an exception
	// because openReadable doens't  validate it and just return a nullptr
	if(directory_uris.size() > 0 ){
		auto fileStatus = BlazingContext::getInstance()->getFileSystemManager()->getFileStatus(this->directory_uris[this->directory_current_file]);

		std::shared_ptr<arrow::io::RandomAccessFile> file =
					BlazingContext::getInstance()->getFileSystemManager()->openReadable(
							this->directory_uris[this->directory_current_file]);

		this->opened_files.push_back(file);

		this->directory_current_file++;
		if(this->directory_current_file > directory_uris.size()){
			this->directory_uris = {};
			this->current_file++;
		}

		return file;
	}else{
		FileStatus fileStatus; 
		auto current_uri = this->file_uris[this->current_file];
		try {
			auto fs_manager = BlazingContext::getInstance()->getFileSystemManager();
			if (fs_manager && fs_manager->exists(current_uri) ) {
				fileStatus = BlazingContext::getInstance()->getFileSystemManager()->getFileStatus(current_uri);
			} else {
				throw std::runtime_error("Path '" + current_uri.toString() +  "' does not exist. Use following template hdfs://hdfs_IP/path_to/filename");	
			}
		}catch (const std::exception & e){
			std::cerr << e.what() << std::endl;
			throw;
		}catch (...) {
			throw;
		}

		std::shared_ptr<arrow::io::RandomAccessFile> file =
				BlazingContext::getInstance()->getFileSystemManager()->openReadable(current_uri);

		fileStatus = BlazingContext::getInstance()->getFileSystemManager()->getFileStatus(current_uri);
		if(fileStatus.isDirectory()){
			this->directory_uris = BlazingContext::getInstance()->getFileSystemManager()->list(current_uri);
			this->directory_current_file = 0;
			return get_next();

		}else{
			this->current_file++;
		}
		this->opened_files.push_back(file);

		return file;

	}

}

std::shared_ptr<arrow::io::RandomAccessFile> uri_data_provider::get_first(){
	// TODO: Take a look at this later, just calling this function to ensure
	// the uri is in a valid state otherwise throw an exception
	// because openReadable doens't  validate it and just return a nullptr
	if(this->file_uris.size() == 0){
		return nullptr;
	}

	std::shared_ptr<arrow::io::RandomAccessFile> file =
			BlazingContext::getInstance()->getFileSystemManager()->openReadable(
					this->file_uris[0]);
	return file;
}

std::vector<std::string> uri_data_provider::get_errors(){
	return this->errors;
}

} /* namespace io */
} /* namespace ral */
