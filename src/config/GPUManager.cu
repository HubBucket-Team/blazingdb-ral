#include <cuda.h>
#include <exception>
#include "GPUManager.cuh"
#include "Utils.cuh"


namespace ral {
namespace config {

GPUManager::GPUManager() : currentDeviceId{0} {
  CheckCudaErrors( cudaGetDeviceCount(&totalDevices) );
}

GPUManager& GPUManager::getInstance() {
  static GPUManager instance;
  return instance;
}

void GPUManager::initialize(int deviceId) {
  if (deviceId < 0 || deviceId >= totalDevices) {
    throw std::runtime_error("In GPUManager::initialize function: Invalid deviceId");
  }

  currentDeviceId = deviceId;
}

void GPUManager::setDevice() {
  CheckCudaErrors( cudaSetDevice(currentDeviceId) );
}

}  // namespace config
}  // namespace ral
