#include "CommunicationData.h"

namespace ral {
namespace communication {
using namespace blazingdb::communication;

CommunicationData::CommunicationData() : orchestratorPort{0} {}

CommunicationData& CommunicationData::getInstance() {
  static CommunicationData communicationData;
  return communicationData;
}

void CommunicationData::initialize(int unixSocketId, const std::string& orchIp,
                                   int16_t orchPort,
                                   const std::string& selfRalIp,
                                   int16_t selfRalPort) {
  orchestratorIp = orchIp;
  orchestratorPort = orchPort;
  selfNode = Node::make(unixSocketId, selfRalIp, selfRalPort);
}

const Node& CommunicationData::getSelfNode() { return *selfNode; }

std::string CommunicationData::getOrchestratorIp() { return orchestratorIp; }

int16_t CommunicationData::getOrchestratorPort() { return orchestratorPort; }

}  // namespace communication
}  // namespace ral