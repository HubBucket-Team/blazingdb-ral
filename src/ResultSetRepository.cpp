/*
 * ResultSetRepository.cpp
 *
 *  Created on: Sep 21, 2018
 *      Author: felipe
 */

#include "ResultSetRepository.h"
#include <random>
#include <algorithm>
#include "cuDF/Allocator.h"

result_set_repository::result_set_repository() {
	// nothing really has to be instantiated

}

result_set_repository::~result_set_repository() {
	//nothing needs to be destroyed
}

void result_set_repository::add_token(query_token_t token, connection_id_t connection){
	std::lock_guard<std::mutex> guard(this->repo_mutex);
	blazing_frame temp;
	this->result_sets[token] = {false, temp, 0.0, "", 0};

	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		std::vector<query_token_t> empty_tokens;
		this->connection_result_sets[connection] = empty_tokens;
	}

	this->connection_result_sets[connection].push_back(token);
}

query_token_t result_set_repository::register_query(connection_id_t connection){
	/*enable this when sessions works! if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}*/

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<query_token_t> dis(
			std::numeric_limits<query_token_t>::min(),
			std::numeric_limits<query_token_t>::max());

	query_token_t token = dis(gen);
	this->add_token(token,connection);
	return token;
}

/*void write_response(blazing_frame frame,response_descriptor response_to_write){
	//TODO: use flatbuffers here to convert the frame to the response message
	//deregister output since we are going to ipc it
	for(size_t i = 0; i < frame.get_width(); i++){
		GDFRefCounter::getInstance()->deregister_column(frame.get_column(i).get_gdf_column());
	}

	//std::lock_guard<std::mutex> guard(this->repo_mutex);
	//TODO: pass in query token and connection id so we can remove these form the map

}*/

void result_set_repository::update_token(query_token_t token, blazing_frame frame, double duration, std::string errorMsg){
	if(this->result_sets.find(token) == this->result_sets.end()){
		throw std::runtime_error{"Token does not exist"};
	}

	//deregister output since we are going to ipc it
	for(size_t i = 0; i < frame.get_width(); i++){
		if(frame.get_column(i).dtype() == GDF_STRING_CATEGORY){
			//we need to convert GDF_STRING_CATEGORY to GDF_STRING
			//for now we can do something hacky lik euse the data pointer to store this

			//TODO the gather_and_remap here is for example in the case of sorting where the order of the indexes changes
			//we must figure out a way to avoid this when is no needed
			NVCategory* new_category = static_cast<NVCategory *> (frame.get_column(i).dtype_info().category)->gather_and_remap( static_cast<int *>(frame.get_column(i).data()), frame.get_column(i).size());
			NVStrings * new_strings = new_category->to_strings();

			gdf_column * new_gdf_column = new gdf_column;
			new_gdf_column->size = frame.get_column(i).size();
			new_gdf_column->null_count = 0;
			new_gdf_column->valid = nullptr;
			new_gdf_column->data = (void * ) new_strings;
			new_gdf_column->dtype = frame.get_column(i).dtype();
			new_gdf_column->col_name = const_cast<char*>(frame.get_column(i).name().c_str());

			gdf_column_cpp string_column;
			string_column.create_gdf_column(new_gdf_column);
			string_column.get_gdf_column()->dtype = GDF_STRING; //TODO create_gdf_column no soporta GDF_STRING porque type_dispatcher tampoco lo soporta
																//esto significa que allocated_size_data tecnicamente esta incorrecto
			frame.set_column(i,string_column);
		}else{
			GDFRefCounter::getInstance()->deregister_column(frame.get_column(i).get_gdf_column());
		}
	}

	for(size_t i = 0; i < frame.get_width(); i++){
		column_token_t column_token = frame.get_column(i).get_column_token();

		if(column_token == 0){
			column_token = gen_token<column_token_t>();
			frame.get_column(i).set_column_token(column_token);
			this->precalculated_columns[column_token] = frame.get_column(i);
		}
	}

	{
		std::lock_guard<std::mutex> guard(this->repo_mutex);
		this->result_sets[token] = {true, frame, duration, errorMsg, 0};
	}
	cv.notify_all();
	/*if(this->requested_responses.find(token) != this->requested_responses.end()){
		write_response(std::get<1>(this->result_sets[token]),this->requested_responses[token]);
	}*/
}

//ToDo uuid instead dummy random
connection_id_t result_set_repository::init_session(){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<connection_id_t> dis(
			std::numeric_limits<connection_id_t>::min(),
			std::numeric_limits<connection_id_t>::max());

	connection_id_t session = dis(gen);

	if(this->connection_result_sets.find(session) != this->connection_result_sets.end()){
		throw std::runtime_error{"Connection already exists"};
	}

	std::lock_guard<std::mutex> guard(this->repo_mutex);
	std::vector<query_token_t> empty_tokens;
	this->connection_result_sets[session] = empty_tokens;
	return session;
}

void result_set_repository::free_result(connection_id_t connection, query_token_t token){
	std::vector<query_token_t>& tokens = this->connection_result_sets[connection];
	tokens.erase(std::remove(tokens.begin(), tokens.end(), token), tokens.end()); //remove

	std::cout<<"freed result!"<<std::endl;

	blazing_frame output_frame = this->result_sets[token].result_frame;

	for(size_t i = 0; i < output_frame.get_width(); i++){
		if(output_frame.get_column(i).dtype() == GDF_STRING){
			NVStrings::destroy(static_cast<NVStrings *>(output_frame.get_column(i).data()));
			output_frame.get_column(i).get_gdf_column()->data = nullptr;
			GDFRefCounter::getInstance()->free(output_frame.get_column(i).get_gdf_column());
		} else if (output_frame.get_column(i).dtype() == GDF_STRING_CATEGORY){
			NVCategory::destroy(static_cast<NVCategory *>(output_frame.get_column(i).dtype_info().category));
			GDFRefCounter::getInstance()->free(output_frame.get_column(i).get_gdf_column());
		}else{
			GDFRefCounter::getInstance()->free(output_frame.get_column(i).get_gdf_column());
		}
	}

	this->result_sets.erase(token);
}

void result_set_repository::remove_all_connection_tokens(connection_id_t connection){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		//TODO percy uncomment this later
		//WARNING uncomment this ... avoid leaks
		//throw std::runtime_error{"Closing a connection that did not exist"};
	}

	std::lock_guard<std::mutex> guard(this->repo_mutex);
	for(query_token_t token : this->connection_result_sets[connection]){
		if(this->result_sets[token].ref_counter == 0){
			this->free_result(connection, token);
		}
	}
	this->connection_result_sets.erase(connection);
}

bool result_set_repository::try_free_result(connection_id_t connection, query_token_t token){
	std::lock_guard<std::mutex> guard(this->repo_mutex);

	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	if(this->result_sets.find(token) != this->result_sets.end()){
		if(this->result_sets[token].ref_counter == 1 ){ //this is the last one reference
			this->free_result(connection, token);
		}
		else{ //it is being referenced yet
			std::cout<<"can't free result, still has at least one reference!"<<std::endl;
			this->result_sets[token].ref_counter--;
		}
		return true;
	}
	return false;
}

result_set_t result_set_repository::get_result(connection_id_t connection, query_token_t token){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	if(this->result_sets.find(token) == this->result_sets.end()){
		throw std::runtime_error{"Result set does not exist"};
	}
	{
		std::cout<<"Result is ready = "<<this->result_sets[token].is_ready<<std::endl;
		//scope the lockguard here
		std::unique_lock<std::mutex> lock(this->repo_mutex);
		cv.wait(lock,[this,token](){
			return this->result_sets[token].is_ready;
		});
		std::cout<<"Result is after lock = "<<this->result_sets[token].is_ready<<std::endl;

		this->result_sets[token].ref_counter++;

		blazing_frame output_frame = this->result_sets[token].result_frame;

		for(size_t i = 0; i < output_frame.get_width(); i++){
			GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		}

		return this->result_sets[token];
	}
}

//WARNING do not call this on anything that will be ipced!!! 
gdf_column_cpp result_set_repository::get_column(connection_id_t connection, column_token_t columnToken){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	if(this->precalculated_columns.find(columnToken) == this->precalculated_columns.end()){
		throw std::runtime_error{"Column does not exist"};
	}
	if(this->precalculated_columns[columnToken].dtype() == GDF_STRING){
		gdf_column_cpp temp_column; //allocar convertir a NVCategory
		NVStrings * strings = static_cast<NVStrings *>(this->precalculated_columns[columnToken].data());
		NVCategory * category = NVCategory::create_from_strings(*strings);
		temp_column.create_gdf_column(GDF_STRING_CATEGORY,this->precalculated_columns[columnToken].size(),nullptr,sizeof(int32_t),this->precalculated_columns[columnToken].name());
		temp_column.get_gdf_column()->dtype_info.category = category;
		return temp_column;
	}else{
		return this->precalculated_columns[columnToken];
	}

}
