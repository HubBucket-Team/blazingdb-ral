/*
 * LogicalFilter.h
 *
 *  Created on: Jul 20, 2018
 *      Author: felipe
 */

#ifndef LOGICALFILTER_H_
#define LOGICALFILTER_H_

#include <vector>
#include <gdf/gdf.h>
#include <string>
#include "DataFrame.h"
#include "Utils.cuh"

gdf_error evaluate_expression(
		blazing_frame inputs,
		std::string expression,
		gdf_column_cpp output,
		gdf_column_cpp temp);


#endif /* LOGICALFILTER_H_ */
