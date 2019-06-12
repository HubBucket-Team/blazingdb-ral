/*
 * DataProvider.h
 *
 *  Created on: Nov 29, 2018
 *      Author: felipe
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <arrow/io/interfaces.h>
#include <memory>
#include <string>
#include <vector>

namespace ral {
namespace io {
/**
 * A class we can use which will be the base for all of our data providers
 */
class data_provider {
public:
	/**
	 * tells us if this provider can generate more arrow::io::RandomAccessFile instances
	 */
	virtual bool has_next() = 0;

	/**
	 *  Resets file read count to 0 for file based DataProvider
	 */
	virtual void reset() = 0;

	/**
	 * gets us the next arrow::io::RandomAccessFile
	 */
	virtual std::shared_ptr<arrow::io::RandomAccessFile> get_next() = 0;
	/**
	 * gets any errors that occured while opening the files
	 */
	virtual std::shared_ptr<arrow::io::RandomAccessFile> get_first() = 0;
	virtual std::vector<std::string> get_errors() = 0;
	virtual std::string get_current_user_readable_file_handle() = 0;

	virtual std::vector<std::shared_ptr<arrow::io::RandomAccessFile> > get_all() = 0;
private:


};

} /* namespace io */
} /* namespace ral */

#endif /* DATAPROVIDER_H_ */
