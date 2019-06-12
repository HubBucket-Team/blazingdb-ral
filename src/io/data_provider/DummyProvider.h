/*
 * uridataprovider.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef DUMMYPROVIDER_H_
#define DUMMYPROVIDER_H_

#include "DataProvider.h"
#include <vector>
#include <arrow/io/interfaces.h>
#include "FileSystem/Uri.h"
#include <memory>


namespace ral {
namespace io {

class dummy_data_provider: public data_provider {
public:
	dummy_data_provider(){

	}

	virtual ~dummy_data_provider(){

	}

	bool has_next(){
		return false;
	}

	void reset(){
		// does nothing
	}

	std::shared_ptr<arrow::io::RandomAccessFile> get_next(){
		return nullptr;
	}

	std::shared_ptr<arrow::io::RandomAccessFile> get_first(){
		return nullptr;
	}


	std::vector<std::string> get_errors(){
		return {};
	}

	std::string get_current_user_readable_file_handle(){
		return "";
	}

	std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > get_all(){
		return {};
	}


	size_t get_file_index(){
		return 0;
	}
private:



};

} /* namespace io */
} /* namespace ral */

#endif /* DUMMYPROVIDER_H_ */
