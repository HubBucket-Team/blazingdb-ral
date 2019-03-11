#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "blazingdb/communication/network/Status.h"
#include "distribution/NodeColumns.h"
#include "distribution/primitives.h"
#include "communication/CommunicationData.h"
#include "communication/factory/MessageFactory.h"
#include "Traits/RuntimeTraits.h"
#include "tests/distribution/mocking/ClientMock.h"
#include "tests/distribution/mocking/ServerMock.h"

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

} // namespace
