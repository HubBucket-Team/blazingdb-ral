#ifndef PARSERUTILS_H_
#define PARSERUTILS_H_

#include "cudf.h"
#include <arrow/io/file.h>

namespace ral {
namespace io {
/**
 * reads contents of an arrow::io::RandomAccessFile in a char * buffer up to the number of bytes specified in bytes_to_read
 * for non local filesystems where latency and availability can be an issue it will ret`ry until it has exhausted its the read attemps and empty reads that are allowed
 */
gdf_error read_file_into_buffer(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                int64_t bytes_to_read,
                                uint8_t* buffer,
                                int total_read_attempts_allowed,
                                int empty_reads_allowed);

} /* namespace io */
} /* namespace ral */

#endif /* PARSERUTILS_H_ */
