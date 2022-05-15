#include "GPUDispatcherInterface.h"

#include <new>

#include "GPUDispatcher.cuh"

#ifdef LOGGER
#define BEGIN_LOG Logger::begin(__func__);
#define BEGIN_LOG Logger::end(__func__);

#define RETURN_GPUDISPATCHER_TASK(FUNC) \
	BEGIN_LOG; \
	auto task = FUNC; \
	END_LOG; \
	return task;
#else
#define BEGIN_LOG
#define END_LOG
#define RETURN_GPUDISPATCHER_TASK(FUNC) return FUNC;
#endif

#define RAISE_ERROR_IF_POINTER_IS_NULL(pointer,errorStatus) \
	if(pointer == nullptr) { \
		std::cout << "Pointer is NULL\n"; \
		*errorStatus = GPUDISPATCHER_ERROR; \
	} else { \
		*errorStatus = GPUDISPATCHER_SUCCESS; \
	} 

void* GPUDispatcher_new(int sizeElementX_in, int sizeElementY_in, int sizeElementZ_in, int sizeType_in, int* errorStatus) {
	int errorStatus_GPU = 0;
	GPUController* controller = new GPUController(sizeElementX_in, sizeElementY_in, sizeElementZ_in, sizeType_in, &errorStatus_GPU);
	RAISE_ERROR_IF_POINTER_IS_NULL(controller, errorStatus);
	return (void*)controller;
}
void GPUDISPATCHER_NEW(FORTRAN_POINTER& pointer, long long int* sizeElementX, long long int* sizeElementY, long long int* sizeElementZ, long long int* sizeType, int* errorStatus) {
	pointer = (FORTRAN_POINTER)GPUDispatcher_new(*sizeElementX, *sizeElementY, *sizeElementZ, *sizeType, errorStatus);
}

void GPUDispatcher_initializeGlobal(void* pointer, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	((GPUController*)pointer)->InitializeGlobal(&errorStatus_GPU);
}
void GPUDISPATCHER_INITIALIZEGLOBAL(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_initializeGlobal((void*)pointer, errorStatus);
}

void GPUDispatcher_initializeLocal(void* pointer, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	((GPUController*)pointer)->InitializeLocal(&errorStatus_GPU);
}
void GPUDISPATCHER_INITIALIZELOCAL(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_initializeLocal((void*)pointer, errorStatus);
}

void GPUDispatcher_loadGlobal(void* pointer, void* pointer_matrix, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer_matrix, errorStatus);
	((GPUController*)pointer)->LoadGlobal(pointer_matrix, &errorStatus_GPU);
}
void GPUDISPATCHER_LOADGLOBAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus) {
	GPUDispatcher_loadGlobal((void*)pointer, (void*)pointer_matrix, errorStatus);
}

void GPUDispatcher_loadLocal(void* pointer, void* pointer_matrix, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer_matrix, errorStatus);
	((GPUController*)pointer)->LoadLocal(pointer_matrix, &errorStatus_GPU);
}
void GPUDISPATCHER_LOADLOCAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus) {
	GPUDispatcher_loadLocal((void*)pointer, (void*)pointer_matrix, errorStatus);
}

void GPUDispatcher_launchTask(void* pointer, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	((GPUController*)pointer)->LaunchTask(&errorStatus_GPU);
}
void GPUDISPATCHER_LAUNCHTASK(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_launchTask((void*)pointer, errorStatus);
}

void GPUDispatcher_unloadLocal(void* pointer, void* pointer_matrix, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer_matrix, errorStatus);
	((GPUController*)pointer)->UnloadLocal(pointer_matrix, &errorStatus_GPU);
}
void GPUDISPATCHER_UNLOADLOCAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus) {
	GPUDispatcher_unloadLocal((void*)pointer, (void*)pointer_matrix, errorStatus);
}

void GPUDispatcher_freeGlobal(void* pointer, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	((GPUController*)pointer)->FreeGlobal(&errorStatus_GPU);
}
void GPUDISPATCHER_FREEGLOBAL(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_freeGlobal((void*)pointer, errorStatus);
}

void GPUDispatcher_freeLocal(void* pointer, int* errorStatus) {
	int errorStatus_GPU = 0;
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	((GPUController*)pointer)->FreeLocal(&errorStatus_GPU);
}
void GPUDISPATCHER_FREELOCAL(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_freeLocal((void*)pointer, errorStatus);
}

void GPUDispatcher_delete(void* pointer, int* errorStatus) {
	RAISE_ERROR_IF_POINTER_IS_NULL(pointer, errorStatus);
	delete (GPUController*)pointer;
}
void GPUDISPATCHER_DELETE(FORTRAN_POINTER& pointer, int* errorStatus) {
	GPUDispatcher_delete((void*)pointer, errorStatus);
}