#include <future>
#include <algorithm>
#include "config/GPUManager.cuh"
#include "operators/JoinOperator.h"
#include "CalciteInterpreter.h"
#include "CodeTimer.h"
#include "ColumnManipulation.cuh"
#include "JoinProcessor.h"
#include "exception/RalException.h"
#include "communication/CommunicationData.h"
#include "distribution/primitives.h"
#include "distribution/NodeColumns.h"
#include "utilities/RalColumn.h"
#include "utilities/TableWrapper.h"
#include "blazingdb/io/Library/Logging/Logger.h"

namespace {
using ral::distribution::NodeColumns;
using blazingdb::communication::Node;
using blazingdb::communication::Context;
} // namespace

const std::string INNER_JOIN = "inner";

namespace ral {
namespace operators {

class JoinOperator {
public:
    JoinOperator(const Context* context);

public:
    virtual blazing_frame operator()(blazing_frame& input, const std::string& query_part) = 0;

protected:
    void evaluate_join(blazing_frame& input, const std::string& query_part);

    void materialize_column(blazing_frame& input, bool is_inner_join);

protected:
    const Context* context_;
    CodeTimer timer_;

protected:
    gdf_column_cpp left_indices_;
    gdf_column_cpp right_indices_;
};


class LocalJoinOperator : public JoinOperator {
public:
    LocalJoinOperator(const Context* context);

public:
    blazing_frame operator()(blazing_frame& input, const std::string& query_part) override;
};


class DistributedJoinOperator : public JoinOperator {
public:
    DistributedJoinOperator(const Context* context);

public:
    blazing_frame operator()(blazing_frame& input, const std::string& query_part) override;

protected:
    blazing_frame process_distribution(blazing_frame& frame, const std::string& query);

    std::vector<gdf_column_cpp> process_distribution_table(std::vector<gdf_column_cpp>& table, std::vector<int>& columnIndices);

    std::vector<gdf_column_cpp> concat_columns(std::vector<gdf_column_cpp>& local_table,
                                               std::vector<NodeColumns>& remote_partition);
};

}  // namespace operators
}  // namespace ral


namespace ral {
namespace operators {

JoinOperator::JoinOperator(const Context* context)
: context_{context} {
    left_indices_ = ral::utilities::create_column(0, ral::traits::dtype<gdf_index_type>);
    right_indices_ = ral::utilities::create_column(0, ral::traits::dtype<gdf_index_type>);
}

//TODO: On error clean up everything here so we dont run out of memory
void JoinOperator::evaluate_join(blazing_frame& input, const std::string& query) {
    std::string condition = get_named_expression(query, "condition");
    std::string join_type = get_named_expression(query, "joinType");

    ::evaluate_join(condition,
                    join_type,
                    input,
                    left_indices_.get_gdf_column(),
                    right_indices_.get_gdf_column());
}


void JoinOperator::materialize_column(blazing_frame& input, bool is_inner_join) {
    std::vector<gdf_column_cpp> new_columns(input.get_size_columns());
    size_t first_table_end_index = input.get_size_column();
    int column_width;
    for(int column_index = 0; column_index < input.get_size_columns(); column_index++){
        gdf_column_cpp output;

        column_width = ral::traits::get_dtype_size_in_bytes(input.get_column(column_index).get_gdf_column());

        if(is_inner_join){
			if (input.get_column(column_index).valid())
				output.create_gdf_column(input.get_column(column_index).dtype(),left_indices_.size(),nullptr,column_width, input.get_column(column_index).name());
			else
				output.create_gdf_column(input.get_column(column_index).dtype(),left_indices_.size(),nullptr,nullptr,column_width, input.get_column(column_index).name());
		} else {
			if (!input.get_column(column_index).valid())
				input.get_column(column_index).allocate_set_valid();

			output.create_gdf_column(input.get_column(column_index).dtype(),left_indices_.size(),nullptr,column_width, input.get_column(column_index).name());
		}

        if(column_index < first_table_end_index)
        {
            //materialize with left_indices
            ::materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),left_indices_.get_gdf_column());
            // std::cout<<"left table output"<<std::endl;
            // print_gdf_column(output.get_gdf_column());
        }else{
            //materialize with right indices
            ::materialize_column(input.get_column(column_index).get_gdf_column(),output.get_gdf_column(),right_indices_.get_gdf_column());
            // std::cout<<"right table output"<<std::endl;
            // print_gdf_column(output.get_gdf_column());
        }

        //TODO: On error clean up all the resources
        //free_gdf_column(input.get_column(column_index));
        output.update_null_count();

        new_columns[column_index] = output;
    }
    input.clear();
    input.add_table(new_columns);
}


LocalJoinOperator::LocalJoinOperator(const Context* context)
: JoinOperator(context)
{ }

blazing_frame LocalJoinOperator::operator()(blazing_frame& input, const std::string& query) {
    // Evaluate join
    timer_.reset();
    evaluate_join(input, query);
    Library::Logging::Logger().logInfo("-> Join sub block 1 took " + std::to_string(timer_.getDuration()) + " ms");

    // Materialize columns
    timer_.reset();
    bool is_inner_join = get_named_expression(query, "joinType") == INNER_JOIN;
    materialize_column(input, is_inner_join);
    Library::Logging::Logger().logInfo("-> Join sub block 2 took " + std::to_string(timer_.getDuration()) + " ms");

    return input;
}


DistributedJoinOperator::DistributedJoinOperator(const Context* context)
: JoinOperator(context)
{ }

blazing_frame DistributedJoinOperator::operator()(blazing_frame& frame, const std::string& query) {
    // Execute distribution
    timer_.reset();
    auto distributed_frame = process_distribution(frame, query);
    Library::Logging::Logger().logInfo("-> Join sub block 0 took " + std::to_string(timer_.getDuration()) + " ms");

    // Evaluate join
    timer_.reset();
    evaluate_join(distributed_frame, query);
    Library::Logging::Logger().logInfo("-> Join sub block 1 took " + std::to_string(timer_.getDuration()) + " ms");

    // Materialize columns
    timer_.reset();
    bool is_inner_join = get_named_expression(query, "joinType") == INNER_JOIN;
    materialize_column(distributed_frame, is_inner_join);
    Library::Logging::Logger().logInfo("-> Join sub block 2 took " + std::to_string(timer_.getDuration()) + " ms");

    // Done
    return distributed_frame;
}

std::vector<gdf_column_cpp> DistributedJoinOperator::process_distribution_table(std::vector<gdf_column_cpp>& table, std::vector<int>& columnIndices) {
    static CodeTimer timer;
    std::vector<NodeColumns> partitions = ral::distribution::generateJoinPartitions(*context_, table, columnIndices);

    distributePartitions(*context_, partitions);
    timer.reset();
    std::vector<NodeColumns> remote_node_columns = ral::distribution::collectPartitions(*context_);
    Library::Logging::Logger().logInfo("-> Join: collectPartitions " + std::to_string(timer.getDuration()) + " ms");

    std::vector<gdf_column_cpp> local_table;
    for (auto& local_node_column : partitions) {
        if (ral::communication::CommunicationData::getInstance().getSelfNode() == local_node_column.getNode()) {
            local_table = local_node_column.getColumns();
            break;
        }
    }
    if (local_table.size() == 0)
        std::cout<<"ERROR: no local_table identified in process_distribution_table"<<std::endl;

    return concat_columns(local_table, remote_node_columns);
}

blazing_frame DistributedJoinOperator::process_distribution(blazing_frame& frame, const std::string& query) {
    std::vector<int> globalColumnIndices;
    parseJoinConditionToColumnIndices(get_named_expression(query, "condition"), globalColumnIndices);

    int processedColumns = 0;
    blazing_frame join_frame;
    for (auto& table : frame.get_columns()) {
        // Get col indices relative to a table, similar to blazing_frame::get_column
        std::vector<int> localIndices;
        std::for_each(globalColumnIndices.begin(), globalColumnIndices.end(), [&](int i){
            if(i >= processedColumns && i < processedColumns + table.size()){
                localIndices.push_back(i - processedColumns);
            }
        } );
        processedColumns += table.size();

        join_frame.add_table(std::move(process_distribution_table(table, localIndices)));
    }

    return join_frame;
}

std::vector<gdf_column_cpp> DistributedJoinOperator::concat_columns(std::vector<gdf_column_cpp>  & local_table,
                                                                    std::vector<NodeColumns>& remote_node_columns) {
    // Get column quantity
    gdf_size_type column_quantity = local_table.size();

    // Concatenate table
    std::vector<gdf_column_cpp> result;
    for (gdf_size_type k = 0; k < column_quantity; ++k) {
        gdf_dtype dtype {GDF_invalid};
        gdf_size_type size{0};
        std::vector<gdf_column*> columns_wrapper;

        if (local_table.size() && local_table[k].size()) {
            size += local_table[k].size();
            dtype = local_table[k].dtype();
            columns_wrapper.emplace_back(local_table[k].get_gdf_column());
        }
        for (auto& remote_node_column : remote_node_columns) {
            auto& remote_columns = remote_node_column.getColumnsRef();
            if (remote_columns.size() == column_quantity && remote_columns.size() && remote_columns[k].get_gdf_column() != nullptr) {
                size += remote_columns[k].size();
                dtype = remote_columns[k].dtype();
                columns_wrapper.emplace_back(remote_columns[k].get_gdf_column());
            }
        }

        // Create output column
        gdf_column_cpp output = ral::utilities::create_column(size, dtype);
        if (size != 0) {
            // Perform concatenation
            CUDF_CALL( gdf_column_concat(output.get_gdf_column(),
                                        columns_wrapper.data(),
                                        columns_wrapper.size()) );
        }
        result.emplace_back(output);
    }

    return result;
}

}  // namespace operators
}  // namespace ral


namespace ral {
namespace operators {

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";

bool is_join(const std::string& query) {
    return (query.find(LOGICAL_JOIN_TEXT) != std::string::npos);
}

blazing_frame process_join(const Context* context, blazing_frame& frame, const std::string& query) {
    std::unique_ptr<JoinOperator> join_operator;
    if (context == nullptr) {
        join_operator = std::make_unique<LocalJoinOperator>(context);
    } else if (context->getTotalNodes() <= 1) {
        join_operator = std::make_unique<LocalJoinOperator>(context);
    }
    else {
        join_operator = std::make_unique<DistributedJoinOperator>(context);
    }
    return (*join_operator)(frame, query);
}

}  // namespace operators
}  // namespace ral
