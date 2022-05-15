#pragma once

#ifndef GPUPDISPATCHER_HEADER
#define GPUDISPATCHER_HEADER


#ifdef WIN32
#ifdef GPUDISPATCHER_EXPORTS
#define GPUDISPATCHER_API __declspec(dllexport)
#else
#define GPUDISPATCHER_API __declspec(dllexport)
#endif
#define FORTRAN_POINTER __int64
#else
#define GPUDISPATCHER_API

#include <cstdint>

#define FORTRAN_POINTER int64_t
#endif

#define GPUDISPATCHER_SUCCESS 0
#define GPUDISPATCHER_ERROR 1

extern "C" {
	//New GPU Controller
	GPUDISPATCHER_API void* GPUDispatcher_new(int sizeElementX_in, int sizeElementY_in, int sizeElementZ_in, int sizeType_in, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_NEW(FORTRAN_POINTER& pointer, long long int* sizeElementX_in, long long int* sizeElementY_in, long long int* sizeElementZ_in, long long int* sizeType_in, int* errorStatus);

	//Initialize Global Memory Allocation (main process per machine)
	GPUDISPATCHER_API void GPUDispatcher_initializeGlobal(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_INITIALIZEGLOBAL(FORTRAN_POINTER& pointer, int* errorStatus);
	
	//Initialize Local Memory Allocation (all processes per machine)
	GPUDISPATCHER_API void GPUDispatcher_initializeLocal(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_INITIALIZELOCAL(FORTRAN_POINTER& pointer, int* errorStatus);

	//Load Global Memory (main process per machine)
	GPUDISPATCHER_API void GPUDispatcher_loadGlobal(void* pointer, void* pointer_matrix, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_LOADGLOBAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus);

	//Load Local Memory (all processes per machine)
	GPUDISPATCHER_API void GPUDispatcher_loadLocal(void* pointer, void* pointer_matrix, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_LOADLOCAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus);

	//Launch Task with Loaded Memory (all processes per machine)
	GPUDISPATCHER_API void GPUDispatcher_launchTask(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_LAUNCHTASK(FORTRAN_POINTER& pointer, int* errorStatus);

	//Unload Local Memory (all processes per machine)
	GPUDISPATCHER_API void GPUDispatcher_unloadLocal(void* pointer, void* pointer_matrix, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_UNLOADLOCAL(FORTRAN_POINTER& pointer, double* pointer_matrix, int* errorStatus);

	//Free Global Memory Allocation (main process per machine)
	GPUDISPATCHER_API void GPUDispatcher_freeGlobal(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_FREEGLOBAL(FORTRAN_POINTER& pointer, int* errorStatus);

	//Free Local Memory Allocation (all processes per machine)
	GPUDISPATCHER_API void GPUDispatcher_freeLocal(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_FREELOCAL(FORTRAN_POINTER& pointer, int* errorStatus);

	//Delete GPU Controller
	GPUDISPATCHER_API void GPUDispatcher_delete(void* pointer, int* errorStatus);
	GPUDISPATCHER_API void GPUDISPATCHER_DELETE(FORTRAN_POINTER& pointer, int* errorStatus);

}


#endif