/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include "binary-operation/jit/util/operator.h"
#include "binary-operation/jit/util/type.h"

namespace gdf {
namespace binops {
namespace jit {

    Operator::Operator()
     : buffer{'\0'}
    { }

    char* Operator::getOperatorName(gdf_binary_operator ope, Operator::Type type) {
        if (type == Operator::Type::Direct) {
            buffer[0] = '\0';
        } else {
            buffer[0] = 'R';
            buffer[1] = '\0';
        }
        strcat(buffer, jit::getOperatorName(ope));
        return buffer;
    }

} // namespace jit
} // namespace binops
} // namespace gdf
