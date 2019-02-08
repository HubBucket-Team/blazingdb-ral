#include <gtest/gtest.h>



#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <gdf_wrapper/gdf_wrapper.cuh>
#include "io/data_parser/CSVParser.h"
#include "io/data_provider/UriDataProvider.h"
#include "io/data_parser/DataParser.h"
#include "io/data_provider/DataProvider.h"

#include <DataFrame.h>
#include <fstream>
#include "CalciteExpressionParsing.h"

#include <blazingdb/io/Library/Logging/Logger.h>
#include <blazingdb/io/Library/Logging/CoutOutput.h>
#include "blazingdb/io/Library/Logging/ServiceLogging.h"

#include <GDFColumn.cuh>


struct ParseCSVTest : public ::testing::Test {
protected:
	//TODO: I copied this from dtypes_test we should put these utils in one place
	//imn guessing it exists but I am not sure where
	template<typename T>
	void Check(gdf_column_cpp out_col, T *host_output) {
		T *device_output;
		device_output = new T[out_col.size()];
		cudaMemcpy(device_output,
				out_col.data(),
				out_col.size() * sizeof(T),
				cudaMemcpyDeviceToHost);

		for (std::size_t i = 0; i < out_col.size(); i++) {
			ASSERT_TRUE(host_output[i] == device_output[i]);
		}
	}


	virtual void SetUp() {
	  	auto output = new Library::Logging::CoutOutput();
  		Library::Logging::ServiceLogging::getInstance().setLogOutput(output);
	}



	template<typename T, typename Functor>
	std::vector<T> get_generated_column(size_t num_rows, size_t column_index,
			Functor & functor){
		std::vector<T> host_column(num_rows);
		for(size_t row_index = 0; row_index  < num_rows; row_index++){
			host_column[row_index] = functor(row_index,column_index);
		}
		return host_column;
	}

};


template<typename Functor>
void generate_csv_file_int32(size_t num_rows, size_t num_cols, std::string path ,Functor & functor){
	//will create a csv file in /tmp folder
	//names will just be col_1 col_2 etc.
	std::ofstream csv_file;
	csv_file.open (path.c_str());


	//iof functor is row_index * ((column_index) * 3);
	//file should look like
	// 0|0|0|0|0
	// 0|3|6|9|12
	// 0|6|12|18|24
	// \n  //@check always last endofline 

	for(size_t row_index = 0; row_index  < num_rows; row_index++){
		if(row_index > 0)	csv_file<<"\n";
		for(size_t column_index = 0; column_index < num_cols; column_index++){
			if(column_index > 0)	csv_file << "|";
			csv_file<< functor (row_index,column_index);
		}
	}
	csv_file<<"\n";
	
	csv_file.close();

}


TEST_F(ParseCSVTest, parse_small_csv_file_int32) {

	{
		size_t num_rows = 10;
		size_t num_cols = 5;
		std::vector<gdf_dtype> types(num_cols,GDF_INT32);
		std::vector<std::string> names = {"a", "b", "c", "d", "e"};

		auto cell_generator= [](size_t row_index, size_t column_index) {
			return (int) (row_index * ((column_index) * 3));
		};


		std::string path = "/tmp/small-test.csv";
		generate_csv_file_int32(
				num_rows,num_cols,path,cell_generator);

		std::vector<std::vector<int> > host_data(num_cols);
		for(size_t column_index = 0; column_index < num_cols; column_index++){
			host_data[column_index] = get_generated_column<int>(num_rows, column_index,cell_generator);
		}

		std::vector<Uri> uris(1);
		uris[0] = Uri(path);
		std::vector<bool> include_column(num_cols,true);
		std::unique_ptr<ral::io::data_provider> provider = std::make_unique<ral::io::uri_data_provider>(uris);
		std::unique_ptr<ral::io::data_parser> parser = std::make_unique<ral::io::csv_parser>("|","\n",0,names,types);


		EXPECT_TRUE(provider->has_next());
		std::vector<gdf_column_cpp> columns;
		parser->parse(provider->get_next(),columns,include_column);

		for(size_t column_index = 0; column_index < num_cols; column_index++){
			Check(columns[column_index], &host_data[column_index][0]);
			
			print_gdf_column(columns[column_index].get_gdf_column());
		}

	}
}


  

template<class FileParserType>
void load_files(FileParserType&& parser, const std::vector<Uri>& uris, std::vector<gdf_column_cpp>& out_columns) {
	auto provider = std::make_unique<ral::io::uri_data_provider>(uris);
	std::vector<std::vector<gdf_column_cpp>> all_parts;
    while (provider->has_next()) {
      std::vector<gdf_column_cpp> columns;
      std::string user_readable_file_handle = provider->get_current_user_readable_file_handle();

      std::shared_ptr<arrow::io::RandomAccessFile> file = provider->get_next();
      if(file != nullptr){
        parser.parse(file, columns);
        all_parts.push_back(columns);
      }else{
        std::cout<<"Was unable to open "<<user_readable_file_handle<<std::endl;
      }
    }
    //checking if any errors occurred
    std::vector<std::string> provider_errors = provider->get_errors();
    if(provider_errors.size() != 0){
      for(size_t error_index = 0; error_index < provider_errors.size(); error_index++){
        std::cout<<provider_errors[error_index]<<std::endl;
      }
    }

    size_t num_files = all_parts.size();
    size_t num_columns = all_parts[0].size();

    if(num_files == 0 || num_columns == 0){ 	//we got no data
      return ;
    }
    if (all_parts.size() == 1) {
        out_columns = all_parts[0];
    } 
    else if (all_parts.size() > 1) {
      std::vector<gdf_column_cpp>& part_left = all_parts[0];
      for(size_t index_col = 0; index_col < part_left.size(); index_col++) { //iterate each one of the columns
        
        std::vector<gdf_column*> columns;
        size_t col_total_size = 0;

        for(size_t index_part = 0; index_part < all_parts.size(); index_part++) { //iterate each one of the parts 
          std::vector<gdf_column_cpp> &part = all_parts[index_part];
          auto &gdf_col = part[index_col];
          columns.push_back(gdf_col.get_gdf_column());
          col_total_size+= gdf_col.size();
        }
        gdf_column_cpp output_col;
        auto & lhs = all_parts[0][index_col];
        output_col.create_gdf_column(lhs.dtype(), col_total_size, nullptr, get_width_dtype(lhs.dtype()), lhs.name());
        gdf_error err = gdf_column_concat(output_col.get_gdf_column(), columns.data(), columns.size());
        if (err == GDF_SUCCESS) {
          out_columns.push_back(output_col);
        } else {
          std::cerr << "ERROR: gdf_column_concat\n";
        }
      }
    }
}

TEST_F(ParseCSVTest, nation_csv) {
	std::cout << "nation_csv\n";
	std::vector<gdf_dtype> types{GDF_INT32, GDF_INT64, GDF_INT32, GDF_INT64};
	std::vector<std::string> names{"n_nationkey", "n_name", "n_regionkey", "n_comment"};
	size_t num_cols = names.size();
	const char* fname = "/tmp/nation.psv";
	std::ofstream outfile(fname, std::ofstream::out);
	auto content = 
R"(0|ALGERIA|0| haggle. carefully final deposits detect slyly agai
1|ARGENTINA|1|al foxes promise slyly according to the regular accounts. bold requests alon
2|BRAZIL|1|y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special 
3|CANADA|1|eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold
4|EGYPT|4|y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d
5|ETHIOPIA|0|ven packages wake quickly. regu
6|FRANCE|3|refully final requests. regular, ironi
7|GERMANY|3|l platelets. regular accounts x-ray: unusual, regular acco
8|INDIA|2|ss excuses cajole slyly across the packages. deposits print aroun
9|INDONESIA|2| slyly express asymptotes. regular deposits haggle slyly. carefully ironic hockey players sleep blithely. carefull
10|IRAN|4|efully alongside of the slyly final dependencies. 
11|IRAQ|4|nic deposits boost atop the quickly final requests? quickly regula
12|JAPAN|2|ously. final, express gifts cajole a
13|JORDAN|4|ic deposits are blithely about the carefully regular pa
14|KENYA|0| pending excuses haggle furiously deposits. pending, express pinto beans wake fluffily past t
15|MOROCCO|0|rns. blithely bold courts among the closely regular packages use furiously bold platelets?
16|MOZAMBIQUE|0|s. ironic, unusual asymptotes wake blithely r
17|PERU|1|platelets. blithely pending dependencies use fluffily across the even pinto beans. carefully silent accoun
18|CHINA|2|c dependencies. furiously express notornis sleep slyly regular accounts. ideas sleep. depos
19|ROMANIA|3|ular asymptotes are about the furious multipliers. express dependencies nag above the ironically ironic account
20|SAUDI ARABIA|4|ts. silent requests haggle. closely express packages sleep across the blithely
21|VIETNAM|2|hely enticingly express accounts. even, final 
22|RUSSIA|3| requests against the platelets use never according to the quickly regular pint
23|UNITED KINGDOM|3|eans boost carefully special requests. accounts are. carefull
24|UNITED STATES|1|y final packages. slow foxes cajole quickly. quickly silent platelets breach ironic accounts. unusual pinto be)";

	outfile <<	content << std::endl;
	outfile.close();

	std::vector<gdf_column_cpp> columns;

	std::vector<Uri> uris(3);
	uris[0] = Uri(fname);
	uris[1] = Uri(fname);
	uris[2] = Uri(fname);

	std::cout << "config provider\n";
	ral::io::uri_data_provider provider(uris);
  std::cout << "\nparsing csv\n\n";

	while(provider.has_next()) {
		std::unique_ptr<ral::io::csv_parser> parser = std::make_unique<ral::io::csv_parser>("|", "\n", 0, names, types);
		parser->parse(provider.get_next(), columns);
		std::cout << "\tnum_cols_out: " << columns.size() << std::endl;
    std::cout << "\tnum_rows_out: " << columns[0].size() << std::endl << std::endl;
	}
  
	ral::io::csv_parser parser("|", "\n", 0, names, types);
	std::vector<gdf_column_cpp> columns_out;
	load_files(std::move(parser), uris, columns_out);
	std::cout << "\nsz: " << columns_out.size() << std::endl;
	
}
