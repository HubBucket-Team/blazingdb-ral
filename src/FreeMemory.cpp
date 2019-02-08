#include "FreeMemory.h"

#include <algorithm>
#include <iostream>
#include <set>

#include <cuda_runtime_api.h>
#include "FreeMemoryPointers.h"


namespace FreeMemory {

void
Initialize() {

}

void
freeAll() {

	FreeMemoryPointers::getInstance().freeAll();

}

void
registerRawPointer(const void *pointer) {
	FreeMemoryPointers::getInstance().registerRawPointer(pointer);
}

void
updateRawPointer(const void *actual, const void *other) {
	//FreeMemoryPointers::getInstance().updateRawPointer(actual, other);
    //std::replace(rawPointers.begin(), rawPointers.end(), actual, other);
}

void
removeRawPointer(const void *pointer) {
	//FreeMemoryPointers::getInstance().updateRawPointer(actual, other).removeRawPointer(pointer);
	//std::remove(rawPointers.begin(), rawPointers.end(), pointer);
}

void
registerIPCPointer(const void *pointer) {
	FreeMemoryPointers::getInstance().registerIPCPointer(pointer);
    //ipcPointers->emplace(pointer);
}

}  // namespace FreeMemory
