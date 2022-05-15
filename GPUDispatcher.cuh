#ifndef GPU_DISPATCHER_HEADER
#define GPU_DISPATCHER_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>

class GPUController {
private:
    void* pointer_GPU;
    void* pointer_CPU;

    void* pointer_Stream_CPU_input;
    void* pointer_Stream_GPU_input;
    void* pointer_Stream_CPU_result;
    void* pointer_Stream_GPU_result;

    int sizeElementX;
    int sizeElementY;
    int sizeElementZ;
    int sizeType;
    int sizeStream;

    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;

    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;

    double* alpha;
    double* beta;

protected:

public:
    //Default Constructor for the GPU Controller class.
    GPUController(int sizeElementsX_in, int sizeElementsY_in, int sizeElementsZ_in, int sizeType_in, int* errorStatus);
    
    //Initialize global memory for general usage (only for the main process per machine).
    void InitializeGlobal(int* errorStatus);

    //Initialize local memory for general usage (for all processes per machine).
    void InitializeLocal(int* errorStatus);
    
    //Load global memory for general usage (only for the main process per machine).
    void LoadGlobal(void* pointer, int* errorStatus);

    //Load local memory for general usage (for all processes per machine).
    void LoadLocal(void* pointer, int* errorStatus);

    //Initialize an asynchronous task.
    void LaunchTask(int* errorStatus);

    //Unload local memory with result (for all processes per machine).
    void UnloadLocal(void* pointer, int* errorStatus);

    //Initialize global memory for general usage (only for the main process per machine).
    void FreeGlobal(int* errorStatus);

    //Initialize local memory for general usage (for all processes per machine).
    void FreeLocal(int* errorStatus);

    //Default Destructor for the GPU Controller class.
    ~GPUController();
};

#endif