#ifndef BLAZINGDB_RAL_GROUPBY_OPERATOR_H
#define BLAZINGDB_RAL_GROUPBY_OPERATOR_H

#include <vector>
#include <string>
#include <blazingdb/communication/Context.h>
#include "DataFrame.h"

namespace ral {
namespace operators {

namespace {
using blazingdb::communication::Context;
} // namespace

bool is_aggregate(std::string query_part);

void process_aggregate(blazing_frame& input, std::string query_part, const Context* queryContext);

}  // namespace operators
}  // namespace ral

#endif  //BLAZINGDB_RAL_GROUPBY_OPERATOR_H
