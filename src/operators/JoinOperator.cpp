#include <future>
#include "operators/JoinOperator.h"
#include "CalciteInterpreter.h"
#include "CodeTimer.h"
#include "ColumnManipulation.cuh"
#include "JoinProcessor.h"
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

namespace ral {
namespace operators {

class JoinOperator {
public:
    JoinOperator(const Context* context);

public:
    virtual blazing_frame operator()(blazing_frame& input, const std::string& query_part) = 0;

protected:
    void evaluate_join(blazing_frame& input, const std::string& query_part);

    void materialize_column(blazing_frame& input);

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
    blazing_frame process_distribution(blazing_frame& input);

    blazing_frame concat_columns(std::vector<NodeColumns>& local_partition,
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


void JoinOperator::materialize_column(blazing_frame& input) {
    std::vector<gdf_column_cpp> new_columns(input.get_size_columns());
    size_t first_table_end_index = input.get_size_column();
    int column_width;
    for(int column_index = 0; column_index < input.get_size_columns(); column_index++){
        gdf_column_cpp output;

        CUDF_CALL( get_column_byte_width(input.get_column(column_index).get_gdf_column(), &column_width) );

        //TODO de donde saco el nombre de la columna aqui???
        output.create_gdf_column(input.get_column(column_index).dtype(),left_indices_.size(),nullptr,column_width, input.get_column(column_index).name());

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
    materialize_column(input);
    Library::Logging::Logger().logInfo("-> Join sub block 2 took " + std::to_string(timer_.getDuration()) + " ms");

    return input;
}


DistributedJoinOperator::DistributedJoinOperator(const Context* context)
: JoinOperator(context)
{ }

blazing_frame DistributedJoinOperator::operator()(blazing_frame& input, const std::string& query) {
    // Execute distribution
    timer_.reset();
    process_distribution(input);
    Library::Logging::Logger().logInfo("-> Join sub block 0 took " + std::to_string(timer_.getDuration()) + " ms");

    // Evaluate join
    timer_.reset();
    evaluate_join(input, query);
    Library::Logging::Logger().logInfo("-> Join sub block 1 took " + std::to_string(timer_.getDuration()) + " ms");

    // Materialize columns
    timer_.reset();
    materialize_column(input);
    Library::Logging::Logger().logInfo("-> Join sub block 2 took " + std::to_string(timer_.getDuration()) + " ms");

    // Done
    return input;
}

blazing_frame DistributedJoinOperator::process_distribution(blazing_frame& input) {
    auto future_node_columns = std::async(std::launch::async,
                                          ral::distribution::collectPartition,
                                          std::ref(*context_));

    // TODO: review whether 'input.get_columns()[0]' is the correct input.
    auto local_node_columns = ral::distribution::generateJoinPartitions(*context_, input.get_columns()[0]);

    distributePartitions(*context_, local_node_columns);

    auto remote_node_columns = future_node_columns.get();

    // TODO: review whether is it necessary to concat columns.
    return concat_columns(local_node_columns, remote_node_columns);
}

// TODO: Improve algorithm
blazing_frame DistributedJoinOperator::concat_columns(std::vector<NodeColumns>& local_node_columns,
                                                      std::vector<NodeColumns>& remote_node_columns) {
    std::vector<std::vector<gdf_column_cpp>> columns;

    // Obtain the current node
    using ral::communication::CommunicationData;
    const Node& node = CommunicationData::getInstance().getSelfNode();

    // Transform local node columns
    for (auto& local_node_column : local_node_columns) {
        if (node == local_node_column.getNode()) {
            auto local_columns = local_node_column.getColumns();
            for (auto local_column : local_columns) {
                std::vector<gdf_column_cpp> vector;
                vector.emplace_back(local_column);
                columns.emplace_back((vector));
            }
        }
    }

    // Erase input data
    local_node_columns.clear();

    // Transform remote node columns
    for (auto& remote_node_column : remote_node_columns) {
        auto remote_columns = remote_node_column.getColumns();
        for (std::size_t k = 0; k < remote_columns.size(); ++k) {
            columns[k].emplace_back((remote_columns[k]));
        }
    }

    // Erase input data
    remote_node_columns.clear();

    // Create result
    std::vector<gdf_column_cpp> result;

    // Concat columns
    for (auto column_vector : columns) {
        // Whether column vector
        if (column_vector.empty()) {
            continue;
        }

        // Get column dtype
        gdf_dtype dtype = column_vector[0].dtype();

        // Calculate total column size
        gdf_size_type size{0};
        for (auto column : column_vector) {
            size += column.size();
        }

        // Create output column
        gdf_column_cpp output = ral::utilities::create_column(size, dtype);

        // Table wrapper input
        ral::utilities::TableWrapper column_split(column_vector);

        // Perform concatenation
        auto error = gdf_column_concat(output.get_gdf_column(),
                                       column_split.getColumns(),
                                       column_split.getQuantity());
        // TODO: Generate Exception
        if (error != GDF_SUCCESS) {
        }

        result.emplace_back(output);
    }

    // Create output
    blazing_frame frame;
    frame.add_table(std::move(result));

    return frame;
}

}  // namespace operators
}  // namespace ral


namespace ral {
namespace operators {

const std::string LOGICAL_JOIN_TEXT = "LogicalJoin";

bool is_join(const std::string& query_part) {
    return (query_part.find(LOGICAL_JOIN_TEXT) != std::string::npos);
}

blazing_frame process_join(const Context* context, blazing_frame& input, const std::string& query_part) {
    std::unique_ptr<JoinOperator> join_processor;
    if (context == nullptr) {
        join_processor = std::make_unique<LocalJoinOperator>(context);
    } else {
        join_processor = std::make_unique<DistributedJoinOperator>(context);
    }
    return (*join_processor)(input, query_part);
}

}  // namespace operators
}  // namespace ral
