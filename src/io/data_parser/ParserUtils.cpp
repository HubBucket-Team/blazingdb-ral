#include <arrow/status.h>
#include "ParserUtils.h"

namespace ral {
namespace io {

gdf_error read_file_into_buffer(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                int64_t bytes_to_read,
                                uint8_t* buffer,
                                int total_read_attempts_allowed,
                                int empty_reads_allowed) {
	if (bytes_to_read > 0){

		int64_t total_read;
		arrow::Status status = file->Read(bytes_to_read,&total_read, buffer);

		if (!status.ok()){
			return GDF_FILE_ERROR;
		}

		if (total_read < bytes_to_read){
			//the following two variables shoudl be explained
			//Certain file systems can timeout like hdfs or nfs,
			//so we shoudl introduce the capacity to retry
			int total_read_attempts = 0;
			int empty_reads = 0;

			while (total_read < bytes_to_read && total_read_attempts < total_read_attempts_allowed && empty_reads < empty_reads_allowed){
				int64_t bytes_read;
				status = file->Read(bytes_to_read-total_read,&bytes_read, buffer + total_read);
				if (!status.ok()){
					return GDF_FILE_ERROR;
				}
				if (bytes_read == 0){
					empty_reads++;
				}
				total_read += bytes_read;
			}
			if (total_read < bytes_to_read){
				return GDF_FILE_ERROR;
			} else {
				return GDF_SUCCESS;
			}
		} else {
			return GDF_SUCCESS;
		}
	} else {
		return GDF_SUCCESS;
	}
}

} /* namespace io */
} /* namespace ral */
