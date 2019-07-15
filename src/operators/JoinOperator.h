#pragma once

#include <string>
#include "DataFrame.h"

// Forward declaration
namespace blazingdb {
namespace communication {
class Context;
}
}

namespace ral {
namespace operators {

// Alias
namespace {
using blazingdb::communication::Context;
}

bool is_join(const std::string& query_part);

blazing_frame process_join(const Context* context, blazing_frame& input, const std::string& query);

}  // namespace operators
}  // namespace ral
