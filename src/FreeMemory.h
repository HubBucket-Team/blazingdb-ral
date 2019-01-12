#ifndef FREE_MEMORY_H_
#define FREE_MEMORY_H_

// TODO: Remove this class.

namespace FreeMemory {
void
Initialize();

void
freeAll();

void
registerRawPointer(const void *);

void
updateRawPointer(const void *actual, const void *other);

void
removeRawPointer(const void *);

void
registerIPCPointer(const void *);

};  // namespace FreeMemory

#endif
