#ifndef COMMUNICATION_DATA_H_
#define COMMUNICATION_DATA_H_

#include <memory>
#include <string>
#include <blazingdb/communication/Node.h>

namespace ral {
namespace communication {

namespace {
    using Node = blazingdb::communication::Node;
}

class CommunicationData
{
public:
  static CommunicationData& getInstance();
  
  void initialize(int unixSocketId, const std::string& orchIp, int16_t orchPort, const std::string& selfRalIp, int16_t selfRalPort);
  const Node& getSelfNode();
  std::string getOrchestratorIp();
  int16_t getOrchestratorPort();

  CommunicationData(CommunicationData&&) = delete;
  CommunicationData(const CommunicationData&) = delete;
  CommunicationData& operator=(CommunicationData&&) = delete;
  CommunicationData& operator=(const CommunicationData&) = delete;

private:
  CommunicationData();

  std::string orchestratorIp;
  int16_t orchestratorPort;
  std::unique_ptr<Node> selfNode;
};

} // namespace communication
} // namespace ral

#endif /* COMMUNICATION_DATA_H_ */