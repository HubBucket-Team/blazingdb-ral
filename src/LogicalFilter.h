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

gdf_error evaluate_expression(
		blazing_frame inputs,
		std::string expression,
		gdf_column * output,
		gdf_column * temp);


#endif /* LOGICALFILTER_H_ */
