#include "FreeMemory.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

static std::vector<const void *> rawPointers;
static std::vector<const void *> ipcPointers;

namespace FreeMemory {
void
freeAll() noexcept {
    std::cout << "\033[32mFreeMemory:\n"
              << "\tRaw: " << rawPointers.size()
              << "\n\tIPC: " << ipcPointers.size() << std::endl;

    std::vector<const void *> pointersToFree;
    std::set_difference(rawPointers.begin(),
                        rawPointers.end(),
                        ipcPointers.begin(),
                        ipcPointers.end(),
                        std::inserter(pointersToFree, pointersToFree.begin()));

    for (auto pointer : pointersToFree) {
        cudaFree(const_cast<void *>(pointer));
    }

    rawPointers.clear();
    ipcPointers.clear();
}

void
registerRawPointer(const void *pointer) {
    rawPointers.push_back(pointer);
}

void
updateRawPointer(const void *actual, const void *other) {
    std::replace(rawPointers.begin(), rawPointers.end(), actual, other);
}

void
removeRawPointer(const void *pointer) {
    std::remove(rawPointers.begin(), rawPointers.end(), pointer);
}

void
registerIPCPointer(const void *pointer) {
    ipcPointers.push_back(pointer);
}

}  // namespace FreeMemory
