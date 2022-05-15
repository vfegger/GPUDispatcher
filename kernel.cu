/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define THREADS 1024


class GPUController {
private:
    void* pointer_GPU;
    void* pointer_CPU;
    int sizeElement;
    int sizeType;

    cudaStream_t* cudaStreams;
    int streamSize;

    void** pointer_Aux_GPU;
    void** result_Aux_GPU;
protected:

public:
    //Default Constructor for the GPU Controller class.
    GPUController() {
        sizeElement = 0;
        sizeType = 0;
        pointer_CPU = NULL;
        pointer_GPU = NULL;
        streamSize = 0;
        cudaStreams = NULL;
        pointer_Aux_GPU = NULL;
        result_Aux_GPU = NULL;
    }

    //Initialize global memory for general usage in any number of streams.
    void InitializeMemory(void* pointer_CPU_in, int sizeElement_in, int sizeType_in) {
        if (pointer_CPU != NULL) {
            cudaFreeHost(pointer_CPU);
        }
        pointer_CPU = NULL;
        if (pointer_GPU != NULL) {
            cudaFree(pointer_GPU);
        }
        pointer_GPU = NULL;
        sizeElement = sizeElement_in;
        sizeType = sizeType_in;
        cudaMallocHost(&pointer_CPU, sizeElement * sizeType);
        cudaMalloc(&pointer_GPU, sizeElement * sizeType);

        //for (unsigned i = 0u; i < sizeElement * sizeType; i++) {
        //    ((char*)pointer_CPU)[i] = ((char*)pointer_CPU_in)[i];
        //}
        cudaMemcpy(pointer_CPU, pointer_CPU_in, sizeElement * sizeType, cudaMemcpyHostToHost);
        cudaMemcpyAsync(pointer_GPU, pointer_CPU, sizeElement * sizeType, cudaMemcpyHostToDevice, cudaStreamDefault);
    }

    //Initialize an array of streams with size 'streamSize_in' and associates each stream with a location in memory for its work. 
    cudaStream_t* InitializeStreams(int streamSize_in) {
        streamSize = streamSize_in;
        cudaStreams = new cudaStream_t[streamSize];
        for (unsigned i = 0u; i < streamSize; i++) {
            cudaStreamCreate(&(cudaStreams[i]));
        }
        pointer_Aux_GPU = new void* [streamSize];
        result_Aux_GPU = new void* [streamSize];
        for (unsigned i = 0; i < streamSize; i++) {
            cudaMalloc(&(pointer_Aux_GPU[i]), sizeElement * sizeType);
            cudaMalloc(&(result_Aux_GPU[i]), sizeElement * sizeType);
        }
        return cudaStreams;
    }

    //Initialize host memory for each stream. It should be used carefully.
    void InitializeTaskMemory(void** aux_CPU, void** result_CPU, int sizeElement, int sizeType) {
        cudaMallocHost(aux_CPU, sizeElement * sizeType);
        cudaMallocHost(result_CPU, sizeElement * sizeType);
    }

    //Initialize an asynchronous task with asynchronous memory copy for a given stream with index cudaStreamIndex and with the given arguments.
    void InitializeTask(unsigned cudaStreamIndex, void* result_CPU, const void* aux_CPU, int sizeElement, int sizeType) {
        dim3 T(THREADS,1,1);
        dim3 B((sizeElement - 1u) / T.x + 1u, 1, 1);

        cudaMemcpyAsync(pointer_Aux_GPU[cudaStreamIndex], aux_CPU, sizeElement * sizeType, cudaMemcpyHostToDevice, cudaStreams[cudaStreamIndex]);
        Task << <B, T, 0, cudaStreams[cudaStreamIndex] >> > ((double*)result_Aux_GPU[cudaStreamIndex], (double*)pointer_GPU, (double*)pointer_Aux_GPU[cudaStreamIndex], sizeElement);
        cudaMemcpyAsync(result_CPU, result_Aux_GPU[cudaStreamIndex], sizeElement * sizeType, cudaMemcpyDeviceToHost, cudaStreams[cudaStreamIndex]);
    }

    //Blocks the host thread until completion of the given stream with index cudaStreamIndex
    void GetTaskOutput(unsigned cudaStreamIndex) {
        cudaStreamSynchronize(cudaStreams[cudaStreamIndex]);
    }

    //Deletes the host memory allocated for a stream. It should be always called before the deletion of the GPU Controller class. It should be used carefully.
    void DeleteTaskMemory(void* aux_CPU, void* result_CPU) {
        cudaFreeHost(result_CPU);
        cudaFreeHost(aux_CPU);
        result_CPU = NULL;
        aux_CPU = NULL;
    }

    //Destroy all streams and the pointer that it was used to hold all of the streams.
    void DestroyStreams() {
        for (unsigned i = 0u; i < streamSize; i++) {
            cudaFree(result_Aux_GPU[i]);
            cudaFree(pointer_Aux_GPU[i]);
            cudaStreamDestroy(cudaStreams[i]);
        }
        delete[] result_Aux_GPU;
        result_Aux_GPU = NULL;
        delete[] pointer_Aux_GPU;
        pointer_Aux_GPU = NULL;
        delete[] cudaStreams;
        cudaStreams = NULL;
    }

    //Default Destructor for the GPU Controller class.
    ~GPUController() {
        if (result_Aux_GPU != NULL) {
            std::cout << "Missed the call to delete ALL streams.";
            delete[] result_Aux_GPU;
        }
        if (pointer_Aux_GPU != NULL) {
            std::cout << "Missed the call to delete ALL streams.";
            delete[] pointer_Aux_GPU;
        }
        if (cudaStreams != NULL) {
            std::cout << "Missed the call to delete ALL streams.";
            delete[] cudaStreams;
        }
        cudaFreeHost(pointer_CPU);
        cudaFree(pointer_GPU);
    }
};

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int i);

int main_args(int argc, char *argv[])
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }
    cudaDeviceReset();
    
    //Start Problem
    const unsigned arraySize = 30u;
    const unsigned streamSize = 20u;
    GPUController* gpuController = new GPUController();
    double* a = new double[arraySize];
    for (unsigned i = 0u; i < arraySize; i++) {
        a[i] = i*1.01;
    }
    gpuController->InitializeMemory(a, arraySize, sizeof(double));

    double** b = new double* [streamSize];
    double** result = new double* [streamSize];
    gpuController->InitializeStreams(streamSize);
    for (unsigned i = 0u; i < streamSize; i++) {
        gpuController->InitializeTaskMemory((void**)&(b[i]), (void**)&(result[i]), arraySize, sizeof(double));
    }
    for (unsigned i = 0u; i < streamSize; i++) {
        for (unsigned j = 0u; j < arraySize; j++) {
            b[i][j] = j+i;
        }
    }


    for (unsigned i = 0u; i < streamSize; i++) {
        gpuController->InitializeTask(i, (void*)result[i], (void*)b[i], arraySize, sizeof(double));
    }
    for (unsigned i = 0u; i < streamSize; i++) {
        gpuController->GetTaskOutput(i);
        for (unsigned j = 0u; j < arraySize; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << "\n";
    }

    for (unsigned i = 0u; i < streamSize; i++) {
        gpuController->DeleteTaskMemory(b[i], result[i]);
    }
    gpuController->DestroyStreams();
    delete gpuController;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/