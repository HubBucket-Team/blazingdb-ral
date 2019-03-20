#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "CalciteExpressionParsing.h"
#include "blazingdb/communication/network/Status.h"
#include "distribution/Exception.h"
#include "distribution/NodeColumns.h"
#include "distribution/NodeSamples.h"
#include "distribution/primitives.h"
#include "communication/CommunicationData.h"
#include "communication/factory/MessageFactory.h"
#include "Traits/RuntimeTraits.h"
#include "utilities/RalColumn.h"
#include "tests/distribution/mocking/ClientMock.h"
#include "tests/distribution/mocking/ServerMock.h"
#include "tests/utilities/gdf_column_cpp_utilities.h"
#include "cuio/parquet/util/bit_util.cuh"

namespace {

// gtest library
using ::testing::_;
using ::testing::ByRef;

// blazingdb communication library
using blazingdb::communication::Node;
using blazingdb::communication::Context;

// Matchers
MATCHER_P(NodeEqual, node, "matcher for node") {
    return arg == node;
}

MATCHER_P(ContextTokenEqual, context_token, "matcher for ContextToken") {
    return arg == context_token;
}

MATCHER_P3(MessageEqual, sender_node, message_token, context_token, "matcher for message") {
    if (arg->getSenderNode() != sender_node) {
        return false;
    }
    if (arg->getMessageTokenValue() != message_token) {
        return false;
    }
    if (arg->getContextTokenValue() != context_token) {
        return false;
    }
    return true;
}

struct DistributionPrimitivesTest : public ::testing::Test {
    DistributionPrimitivesTest() {
        initSelfNode();
        createContext();
        initializeCommunicationData();
    }

    ~DistributionPrimitivesTest() {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }

    void initSelfNode() {
        self_node_ = Node::makeShared(0, self_ip_, self_port_);
    }

    void createContext() {
        // Create master node
        master_node_ = Node::makeShared(12, "192.168.0.2", 1012);

        // Create worker nodes
        std::vector<std::shared_ptr<Node>> nodes;
        nodes.push_back(master_node_);
        nodes.push_back(self_node_);
        nodes.push_back(Node::makeShared(13, "192.168.0.3", 1013));
        nodes.push_back(Node::makeShared(14, "192.168.0.4", 1014));

        // Create logical plan
        std::string logical_plan {"some logical plan"};

        // Create context
        context_ = std::make_shared<Context>(nodes, master_node_, logical_plan);
    }

    void initializeCommunicationData() {
        using ral::communication::CommunicationData;
        auto& communication_data = CommunicationData::getInstance();
        communication_data.initialize(0, "192.168.0.100", 1000, self_ip_, self_port_);
    }

    std::shared_ptr<Context> context_;
    std::shared_ptr<Node> master_node_;

    const std::string self_ip_{"192.168.0.10"};
    const unsigned short self_port_{7897};
    std::shared_ptr<Node> self_node_;
};


TEST_F(DistributionPrimitivesTest, sendSamplesToMasterTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    using ral::communication::messages::SampleToNodeMasterMessage;
    const auto& message_token = SampleToNodeMasterMessage::getMessageID();
    const auto  context_token = context_->getContextToken().getIntToken();

    // Verify test
    EXPECT_CALL(ClientMock::getInstance(),
                send(NodeEqual(*master_node_),
                     MessageEqual(*self_node_, message_token, context_token)))
                .Times(1);

    // Execute test
    std::size_t total_row_size = 1000;
    ral::distribution::sendSamplesToMaster(*context_, std::move(test_columns), total_row_size);
}


TEST_F(DistributionPrimitivesTest, collectPartitionTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createColumnDataMessage(context_token, *self_node_, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(ContextTokenEqual(ByRef(context_token))))
               .Times(context_->getAllNodes().size() - 1)
               .WillRepeatedly(::testing::Return(message));

    // Execute Test
    ral::distribution::collectPartition(*context_);
}


TEST_F(DistributionPrimitivesTest, collectPartitionExceptionTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createSampleToNodeMaster(context_token, *self_node_, 10, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(ContextTokenEqual(ByRef(context_token))))
                .Times(1)
                .WillOnce(::testing::Return(message));

    // Execute Test
    using ral::distribution::MessageMismatchException;
    ASSERT_THROW(ral::distribution::collectPartition(*context_), MessageMismatchException);
}


TEST_F(DistributionPrimitivesTest, collectSamplesTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    std::size_t total_row_size = 1000;
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createSampleToNodeMaster(context_token, *self_node_, total_row_size, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(_))
                .Times(context_->getAllNodes().size() - 1)
                .WillRepeatedly(::testing::Return(message));

    // Execute Test
    ral::distribution::collectSamples(*context_);
}


TEST_F(DistributionPrimitivesTest, collectSamplesExceptionTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createColumnDataMessage(context_token, *self_node_, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(_))
                .Times(1)
                .WillOnce(::testing::Return(message));

    // Execute Test
    using ral::distribution::MessageMismatchException;
    ASSERT_THROW(ral::distribution::collectSamples(*context_), MessageMismatchException);
}


TEST_F(DistributionPrimitivesTest, distributePartitionPlanTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    using ral::communication::messages::ColumnDataMessage;
    const auto& message_token = ColumnDataMessage::getMessageID();
    const auto  context_token = context_->getContextToken().getIntToken();

    // Verify test
    for (auto& node : context_->getWorkerNodes()) {
        EXPECT_CALL(ClientMock::getInstance(),
                    send(NodeEqual(*node),
                         MessageEqual(*self_node_, message_token, context_token)))
                    .Times(1);
    }

    // Execute test
    ral::distribution::distributePartitionPlan(*context_, test_columns);
}


TEST_F(DistributionPrimitivesTest, getPartitionPlanTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createColumnDataMessage(context_token, *self_node_, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(ContextTokenEqual(ByRef(context_token))))
                .Times(1)
                .WillOnce(::testing::Return(message));

    // Execute Test
    ral::distribution::getPartitionPlan(*context_);
}


TEST_F(DistributionPrimitivesTest, getPartitionPlanExceptionTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Get data for validation
    const auto& context_token = context_->getContextToken();

    // Create data - message
    using MessageFactory = ral::communication::messages::Factory;
    auto message = MessageFactory::createSampleToNodeMaster(context_token, *self_node_, 1000, test_columns);

    // Verify test
    EXPECT_CALL(ServerMock::getInstance(), getMessage(ContextTokenEqual(ByRef(context_token))))
                .Times(1)
                .WillOnce(::testing::Return(message));

    // Execute Test
    using ral::distribution::MessageMismatchException;
    ASSERT_THROW(ral::distribution::getPartitionPlan(*context_), MessageMismatchException);
}


TEST_F(DistributionPrimitivesTest, partitionDataTest) {
    // data type
    using DATA_TYPE = std::int32_t;

    // create data - gdf_column_cpp
    std::vector<gdf_column_cpp> table;

    {
        std::vector<DATA_TYPE> data{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        table.emplace_back(ral::utilities::create_column<DATA_TYPE>(data, GDF_INT32));
    }

    // create pivots - gdf_column_cpp
    std::vector<gdf_column_cpp> pivots;

    {
        std::vector<DATA_TYPE> data{ 13, 15, 18 };
        pivots.emplace_back(ral::utilities::create_column<DATA_TYPE>(data, GDF_INT32));
    }

    // output data
    std::vector<std::vector<DATA_TYPE>> output_result = { {10, 11, 12 },
                                                          {15, 16, 17},
                                                          {18, 19} };

    // execute distributed function
    std::vector<ral::distribution::NodeColumns> columns;
    ASSERT_NO_THROW(columns = ral::distribution::partitionData(*context_, table, pivots));

    // verify
    ASSERT_EQ(columns.size(), output_result.size());

    for (std::size_t k = 0; k < columns.size(); ++k) {
        std::vector<gdf_column_cpp> node_columns = columns[k].getColumns();

        ASSERT_EQ(node_columns.size(), 1);

        std::vector<DATA_TYPE> output_data = ral::test::get_column_data<DATA_TYPE>(node_columns[0]);
        ASSERT_EQ(output_data.size(), output_result[k].size());
        for (std::size_t i = 0; i < output_data.size(); ++i) {
            ASSERT_EQ(output_data[i], output_result[k][i]);
        }
    }
}


TEST_F(DistributionPrimitivesTest, distributePartitionsTest) {
    // Create data - gdf_column_cpp
    std::vector<gdf_column_cpp> test_columns;

    // Create data - NodeColumns
    using ral::distribution::NodeColumns;
    std::vector<NodeColumns> node_columns;
    for (const auto node : context_->getWorkerNodes()) {
        node_columns.emplace_back(*node, std::vector<gdf_column_cpp>());
    }

    // Get data for validation
    using ral::communication::messages::ColumnDataMessage;
    const auto& message_token = ColumnDataMessage::getMessageID();
    const auto  context_token = context_->getContextToken().getIntToken();

    // Verify test
    for (auto& node : context_->getWorkerNodes()) {
        EXPECT_CALL(ClientMock::getInstance(),
                    send(NodeEqual(*node),
                         MessageEqual(*self_node_, message_token, context_token)))
                    .Times(1);
    }

    // Execute test
    ral::distribution::distributePartitions(*context_, node_columns);
}


struct GeneratePartitionPlansTest : public ::testing::Test {
    GeneratePartitionPlansTest() {
    }

    ~GeneratePartitionPlansTest() {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }

    std::vector<gdf_valid_type> boolVectorToBitVector(const std::vector<uint8_t>& input){
        std::vector<gdf_valid_type> bitVector;
        bitVector.resize(ral::traits::get_bitmask_size_in_bytes(input.size()));

        auto* bitDataPtr = bitVector.data();
        for(size_t i = 0; i < input.size(); i++)
        {
            if(input[i]){
                gdf::util::turn_bit_on(bitDataPtr, i);
            } else {
                gdf::util::turn_bit_off(bitDataPtr, i);
            }
        }

        return bitVector;
    }
};

TEST_F(GeneratePartitionPlansTest, SingleColumn) {
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(Node::makeShared(13, "192.168.0.3", 1013));
    nodes.push_back(Node::makeShared(14, "192.168.0.4", 1014));

    // Create context
    std::shared_ptr<Context> context = std::make_shared<Context>(nodes, nodes[0], "logical_plan");

    // Create data - NodeSamples
    std::vector<gdf_column_cpp> columnsSample1{1};
    std::vector<int> colData1 = {27, 5, 13, 30, 16};
    columnsSample1[0].create_gdf_column(GDF_INT32, colData1.size(), colData1.data(), get_width_dtype(GDF_INT32), "");

    std::vector<gdf_column_cpp> columnsSample2{1};
    std::vector<int>  colData2 = {11, 41, 1, 3};
    columnsSample2[0].create_gdf_column(GDF_INT32, colData2.size(), colData2.data(), get_width_dtype(GDF_INT32), "");

    using ral::distribution::NodeSamples;
    std::vector<NodeSamples> node_samples;
    node_samples.emplace_back(15, *nodes[0], std::move(columnsSample1));
    node_samples.emplace_back(12, *nodes[1], std::move(columnsSample2));

    std::vector<int8_t> sortOrderTypes = {0};

    // Execute test
    auto pivots = ral::distribution::generatePartitionPlans(*context, node_samples, sortOrderTypes);

    std::vector<std::vector<int>> result = {{13}};
    ASSERT_EQ(result.size(), pivots.size());
    for(std::size_t i = 0; i < pivots.size(); ++i)
    {
        auto& p = pivots[i];
        ASSERT_EQ(result[i].size(), p.size());

        auto dataByteArr = ral::test::get_column_data(p.get_gdf_column());
        int* dataPtr = reinterpret_cast<int*>(dataByteArr.data());
        for (std::size_t j = 0; j < p.size(); ++j) {
            ASSERT_EQ(result[i][j], dataPtr[j]);
        }
    }
}

TEST_F(GeneratePartitionPlansTest, MultiColumn) {
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(Node::makeShared(13, "192.168.0.3", 1013));
    nodes.push_back(Node::makeShared(14, "192.168.0.4", 1014));

    // Create context
    std::shared_ptr<Context> context = std::make_shared<Context>(nodes, nodes[0], "logical_plan");

    // Create data - NodeSamples
    std::vector<std::vector<int>> colData1 = {{27, 5, 13, 30, 16}, {711, 121, 7498, 2866, 7638}};
    std::vector<gdf_column_cpp> columnsSample1{colData1.size()};
    columnsSample1[0].create_gdf_column(GDF_INT32, colData1[0].size(), colData1[0].data(), get_width_dtype(GDF_INT32), "");
    columnsSample1[1].create_gdf_column(GDF_INT32, colData1[1].size(), colData1[1].data(), get_width_dtype(GDF_INT32), "");

    std::vector<std::vector<int>> colData2 = {{11, 41, 1, 3}, {4643, 1182, 2182, 5500}};
    std::vector<gdf_column_cpp> columnsSample2{colData2.size()};
    columnsSample2[0].create_gdf_column(GDF_INT32, colData2[0].size(), colData2[0].data(), get_width_dtype(GDF_INT32), "");
    columnsSample2[1].create_gdf_column(GDF_INT32, colData2[1].size(), colData2[1].data(), get_width_dtype(GDF_INT32), "");

    using ral::distribution::NodeSamples;
    std::vector<NodeSamples> node_samples;
    node_samples.emplace_back(15, *nodes[0], std::move(columnsSample1));
    node_samples.emplace_back(12, *nodes[1], std::move(columnsSample2));

    std::vector<int8_t> sortOrderTypes = {0, 1};

    // Execute test
    auto pivots = ral::distribution::generatePartitionPlans(*context, node_samples, sortOrderTypes);

    std::vector<std::vector<int>> result = {{13}, {7498}};
    std::vector<std::vector<uint8_t>> resultvalids = {{1}, {1}};
    ASSERT_EQ(result.size(), pivots.size());
    for(std::size_t i = 0; i < pivots.size(); ++i)
    {
        auto& p = pivots[i];
        ASSERT_EQ(result[i].size(), p.size());

        auto dataByteArr = ral::test::get_column_data(p.get_gdf_column());
        int* dataPtr = reinterpret_cast<int*>(dataByteArr.data());
        auto validByteArr = ral::test::get_column_valid(p.get_gdf_column());
        uint8_t* validPtr = validByteArr.data();
        for (std::size_t j = 0; j < p.size(); ++j) {
            ASSERT_EQ(resultvalids[i][j], gdf::util::is_valid(validPtr, j));
            if (resultvalids[i][j]) {
                ASSERT_EQ(result[i][j], dataPtr[j]);
            }
        }
    }
}

TEST_F(GeneratePartitionPlansTest, SingleColumnWithNulls) {
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(Node::makeShared(13, "192.168.0.3", 1013));
    nodes.push_back(Node::makeShared(14, "192.168.0.4", 1014));

    // Create context
    std::shared_ptr<Context> context = std::make_shared<Context>(nodes, nodes[0], "logical_plan");

    // Create data - NodeSamples
    std::vector<gdf_column_cpp> columnsSample1{1};
    std::vector<int> colData1 = {27, 5, 13, 30, 16, 13, 1,  12, 17, 7};
    std::vector<uint8_t> valids1 = {1, 0, 1, 0, 1, 1, 1, 1, 1, 0};
    columnsSample1[0].create_gdf_column(GDF_INT32, colData1.size(), colData1.data(), boolVectorToBitVector(valids1).data(), get_width_dtype(GDF_INT32), "");

    std::vector<gdf_column_cpp> columnsSample2{1};
    std::vector<int>  colData2 = {11, 41, 1, 3,  16, 24, 18, 7};
    std::vector<uint8_t> valids2 = {1, 1, 1, 1, 0, 0, 1, 1};
    columnsSample2[0].create_gdf_column(GDF_INT32, colData2.size(), colData2.data(), boolVectorToBitVector(valids2).data(), get_width_dtype(GDF_INT32), "");

    using ral::distribution::NodeSamples;
    std::vector<NodeSamples> node_samples;
    node_samples.emplace_back(30, *nodes[0], std::move(columnsSample1));
    node_samples.emplace_back(24, *nodes[1], std::move(columnsSample2));

    std::vector<int8_t> sortOrderTypes = {0};

    // Execute test
    auto pivots = ral::distribution::generatePartitionPlans(*context, node_samples, sortOrderTypes);

    std::vector<std::vector<int>> result = {{17}};
    ASSERT_EQ(result.size(), pivots.size());
    for(std::size_t i = 0; i < pivots.size(); ++i)
    {
        auto& p = pivots[i];
        ASSERT_EQ(result[i].size(), p.size());

        auto dataByteArr = ral::test::get_column_data(p.get_gdf_column());
        int* dataPtr = reinterpret_cast<int*>(dataByteArr.data());
        for (std::size_t j = 0; j < p.size(); ++j) {
            ASSERT_EQ(result[i][j], dataPtr[j]);
        }
    }
}

TEST_F(GeneratePartitionPlansTest, MultiColumnWithNull) {
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(Node::makeShared(13, "192.168.0.3", 1013));
    nodes.push_back(Node::makeShared(14, "192.168.0.4", 1014));

    // Create context
    std::shared_ptr<Context> context = std::make_shared<Context>(nodes, nodes[0], "logical_plan");

    // Create data - NodeSamples
    std::vector<std::vector<int>> colData1 = {{27, 5, 13, 30, 16, 13, 1,  12, 17, 7}, {711, 121, 7498, 2866, 7638, 8166, 6853, 1709, 4867, -611}};
    std::vector<std::vector<uint8_t>> valids1 = {{1, 0, 1, 0, 1, 1, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
    std::vector<gdf_column_cpp> columnsSample1{colData1.size()};
    columnsSample1[0].create_gdf_column(GDF_INT32, colData1[0].size(), colData1[0].data(), boolVectorToBitVector(valids1[0]).data(), get_width_dtype(GDF_INT32), "");
    columnsSample1[1].create_gdf_column(GDF_INT32, colData1[1].size(), colData1[1].data(), boolVectorToBitVector(valids1[1]).data(), get_width_dtype(GDF_INT32), "");

    std::vector<std::vector<int>> colData2 = {{11, 41, 1, 3,  16, 24, 18, 7}, {4643, 1182, 2182, 5500, -716, 7462, 6505, 2953}};
    std::vector<std::vector<uint8_t>> valids2 = {{1, 1, 1, 1, 0, 0, 1, 1}, {0, 1, 1, 1, 1, 1, 1, 1}};
    std::vector<gdf_column_cpp> columnsSample2{colData2.size()};
    columnsSample2[0].create_gdf_column(GDF_INT32, colData2[0].size(), colData2[0].data(), boolVectorToBitVector(valids2[0]).data(), get_width_dtype(GDF_INT32), "");
    columnsSample2[1].create_gdf_column(GDF_INT32, colData2[1].size(), colData2[1].data(), boolVectorToBitVector(valids2[1]).data(), get_width_dtype(GDF_INT32), "");

    using ral::distribution::NodeSamples;
    std::vector<NodeSamples> node_samples;
    node_samples.emplace_back(15, *nodes[0], std::move(columnsSample1));
    node_samples.emplace_back(12, *nodes[1], std::move(columnsSample2));

    std::vector<int8_t> sortOrderTypes = {0, 1};

    // Execute test
    auto pivots = ral::distribution::generatePartitionPlans(*context, node_samples, sortOrderTypes);

    std::vector<std::vector<int>> result = {{17}, {7498}};
    std::vector<std::vector<uint8_t>> resultvalids = {{1}, {0}};
    ASSERT_EQ(result.size(), pivots.size());
    for(std::size_t i = 0; i < pivots.size(); ++i)
    {
        auto& p = pivots[i];
        ASSERT_EQ(result[i].size(), p.size());

        auto dataByteArr = ral::test::get_column_data(p.get_gdf_column());
        int* dataPtr = reinterpret_cast<int*>(dataByteArr.data());
        auto validByteArr = ral::test::get_column_valid(p.get_gdf_column());
        uint8_t* validPtr = validByteArr.data();
        for (std::size_t j = 0; j < p.size(); ++j) {
            ASSERT_EQ(resultvalids[i][j], gdf::util::is_valid(validPtr, j));
            if (resultvalids[i][j]) {
                ASSERT_EQ(result[i][j], dataPtr[j]);
            }
        }
    }
}

} // namespace
