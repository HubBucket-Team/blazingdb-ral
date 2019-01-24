/*
 * ResultSetRepository.cpp
 *
 *  Created on: Sep 21, 2018
 *      Author: felipe
 */

#include "ResultSetRepository.h"
#include <random>
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
	this->result_sets[token] = std::make_tuple(false, temp, 0.0);

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

void result_set_repository::update_token(query_token_t token, blazing_frame frame, double duration){
	if(this->result_sets.find(token) == this->result_sets.end()){
		throw std::runtime_error{"Token does not exist"};
	}

	//deregister output since we are going to ipc it
	for(size_t i = 0; i < frame.get_width(); i++){
		GDFRefCounter::getInstance()->deregister_column(frame.get_column(i).get_gdf_column());
	}

	{
		std::lock_guard<std::mutex> guard(this->repo_mutex);
		this->result_sets[token] = std::make_tuple(true, frame, duration);
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

void result_set_repository::remove_all_connection_tokens(connection_id_t connection){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		//TODO percy uncomment this later
		//WARNING uncomment this ... avoid leaks
		//throw std::runtime_error{"Closing a connection that did not exist"};
	}

	std::lock_guard<std::mutex> guard(this->repo_mutex);
	for(query_token_t token : this->connection_result_sets[connection]){
		this->result_sets.erase(token);
	}
	this->connection_result_sets.erase(connection);
}

bool result_set_repository::free_result(connection_id_t connection, query_token_t token){
	std::lock_guard<std::mutex> guard(this->repo_mutex);

	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	for(size_t i = 0; i <  this->connection_result_sets[connection].size(); i++){
		query_token_t token_test  = this->connection_result_sets[connection][i];
		if(token == token_test){
			std::vector<query_token_t> tokens= this->connection_result_sets[connection];
			tokens.erase(tokens.begin() + i);
			this->connection_result_sets[connection] = tokens;
			std::cout<<"freed result!"<<std::endl;

			blazing_frame output_frame = std::get<1>(this->result_sets[token]);

			for(size_t i = 0; i < output_frame.get_width(); i++){
				GDFRefCounter::getInstance()->free(output_frame.get_column(i).get_gdf_column());
			}

			this->result_sets.erase(token);
			return true;
		}
	}	
	return false;
}

std::tuple<blazing_frame, double> result_set_repository::get_result(connection_id_t connection, query_token_t token){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	if(this->result_sets.find(token) == this->result_sets.end()){
		throw std::runtime_error{"Result set does not exist"};
	}
	{
		std::cout<<"Result is ready = "<<std::get<0>(this->result_sets[token])<<std::endl;
		//scope the lockguard here
		std::unique_lock<std::mutex> lock(this->repo_mutex);
		cv.wait(lock,[this,token](){
			return std::get<0>(this->result_sets[token]);
		});
		std::cout<<"Result is after lock = "<<std::get<0>(this->result_sets[token])<<std::endl;

		blazing_frame output_frame = std::get<1>(this->result_sets[token]);

		for(size_t i = 0; i < output_frame.get_width(); i++){
			GDFRefCounter::getInstance()->deregister_column(output_frame.get_column(i).get_gdf_column());
		}

		return std::make_tuple(output_frame, std::get<2>(this->result_sets[token]));
	}
}

gdf_column_cpp result_set_repository::get_column(connection_id_t connection, column_token_t columnToken){
	if(this->connection_result_sets.find(connection) == this->connection_result_sets.end()){
		throw std::runtime_error{"Connection does not exist"};
	}

	if(this->precalculated_columns.find(columnToken) == this->precalculated_columns.end()){
		throw std::runtime_error{"Column does not exist"};
	}

	return this->precalculated_columns[columnToken];
}

