/*
 * uridataprovider.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef URIDATAPROVIDER_H_
#define URIDATAPROVIDER_H_

#include "DataProvider.h"
#include <vector>
#include <arrow/io/interfaces.h>
#include "FileSystem/Uri.h"
#include <memory>


namespace ral {
namespace io {
/**
 * can generate a series of randomaccessfiles from uris that are provided
 * when it goes out of scope it will close any files it opened
 * this last point is debatable in terms of if this is the desired functionality
 */
class uri_data_provider: public data_provider {
public:
	uri_data_provider(std::vector<Uri> uris);
	virtual ~uri_data_provider();
	/**
	 * tells us if there are more files in the list of uris to be provided
	 */
	bool has_next();
	/**
	 * gets a randomaccessfile to the uri at file_uris[current_file] and advances current_file by 1
	 */
	std::shared_ptr<arrow::io::RandomAccessFile> get_next();
	/**
	 * returns any errors that were encountered when opening arrow::io::RandomAccessFile
	 */
	std::vector<std::string> get_errors();
	/**
	 * returns a string that the user should be able to use to identify which file is being referred to in error messages
	 */
	std::string get_current_user_readable_file_handle();
private:
	/**
	 * stores the list of uris that will be used by the provider
	 */
	std::vector<Uri> file_uris;
	/**
	 * stores an index to the current file being used
	 */
	size_t current_file;
	/**
	 * stores the files that were opened by the provider to be closed when it goes out of scope
	 */
	std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > opened_files;
	//TODO: we should really be either handling exceptions up the call stack or
	//storing something more elegant than just a string with an error message
	/**
	 * stores any errors that occured while trying to open these uris
	 */
	std::vector<std::string> errors;


};

} /* namespace io */
} /* namespace ral */

#endif /* URIDATAPROVIDER_H_ */
