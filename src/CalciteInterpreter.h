
#ifndef CALCITEINTERPRETER_H_
#define CALCITEINTERPRETER_H_

#include <vector>
#include <gdf/gdf.h>
#include <string>


gdf_error evaluate_query(
		std::vector<std::vector<gdf_column *> > input_tables,
		std::string query,
		std::vector<gdf_column *> outputs,
		void * temp_space);


#endif /* CALCITEINTERPRETER_H_ */
