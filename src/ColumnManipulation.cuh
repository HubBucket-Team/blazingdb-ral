/*
 * ColumnManipulation.cuh
 *
 *  Created on: Aug 9, 2018
 *      Author: felipe
 */

#ifndef COLUMNMANIPULATION_CUH_
#define COLUMNMANIPULATION_CUH_

#include <gdf/gdf.h>

//TODO: in theory  we want to get rid of this
// we should be using permutation iterators when we can
gdf_error materialize_column(gdf_column * input,
		gdf_column * output,
		gdf_column * row_indeces);


#endif /* COLUMNMANIPULATION_CUH_ */
