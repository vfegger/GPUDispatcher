#include "GPUDispatcher.cuh"

#define ERRORLOG

#ifdef ERRORLOG
#define RAISE_CUDA_ERROR(cudaStatus,errorStatus) \
    if(*errorStatus == 0){ \
        *errorStatus = (int)cublasStatus; \
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) { \
            std::cout << "Error Cuda #" << cudaStatus << " : Cannot proceed with the implementation\n"; \
        } \
    }
#define RAISE_CUBLAS_ERROR(cublasStatus,errorStatus) \
    if(*errorStatus == 0){ \
        *errorStatus = (int)cublasStatus; \
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) { \
            std::cout << "Error Cublas #" << cublasStatus << " : Cannot proceed with the implementation\n"; \
        } \
    }
#else
#define RAISE_CUDA_ERROR(cudaStatus,errorStatus)
#define RAISE_CUBLAS_ERROR(cublasStatus,errorStatus)
#endif

GPUController::GPUController(int sizeElementsX_in, int sizeElementsY_in, int sizeElementsZ_in, int sizeType_in, int* errorStatus) {
    sizeElementX = sizeElementsX_in;
    sizeElementY = sizeElementsY_in;
    sizeElementZ = sizeElementsZ_in;
    sizeType = sizeType_in;

    sizeStream = 0;
    pointer_CPU = NULL;
    pointer_GPU = NULL;
    pointer_Stream_CPU_input = NULL;
    pointer_Stream_GPU_input = NULL;
    pointer_Stream_CPU_result = NULL;
    pointer_Stream_GPU_result = NULL;
    cudaStatus = cudaSuccess;
    cublasStatus = CUBLAS_STATUS_SUCCESS;
    
    cudaStatus = cudaSetDevice(0);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaDeviceReset();
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);

    cudaStatus = cudaStreamCreate(&cudaStream);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cublasStatus = cublasCreate_v2(&cublasHandle);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus);
    cublasStatus = cublasSetStream_v2(cublasHandle, cudaStream);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus);

    alpha = new double(1.0);
    beta = new double(0.0);
}

void GPUController::InitializeGlobal(int* errorStatus) {
    cudaStatus = cudaMallocHost(&pointer_CPU, sizeElementX * sizeElementY * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaMalloc(&pointer_GPU, sizeElementX * sizeElementY * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
}

void GPUController::InitializeLocal(int* errorStatus) {
    cudaStatus = cudaMallocHost(&pointer_Stream_CPU_input, sizeElementY * sizeElementZ * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaMallocHost(&pointer_Stream_CPU_result, sizeElementX * sizeElementZ * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaMalloc(&pointer_Stream_GPU_input, sizeElementY * sizeElementZ * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaMalloc(&pointer_Stream_GPU_result, sizeElementX * sizeElementZ * sizeType);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
}

void GPUController::LoadGlobal(void* pointer, int* errorStatus) {
    unsigned size = sizeElementX * sizeElementY;
    if (sizeType == 8) {
        for (unsigned i = 0u; i < size; i++) {
            ((double*)pointer_CPU)[i] = ((double*)pointer)[i];
        }
    }
    else {
        *errorStatus = 1;
        return;
    }
    cublasStatus = cublasSetMatrixAsync(sizeElementX, sizeElementY, sizeType, pointer_CPU, sizeElementX, pointer_GPU, sizeElementX, cudaStream);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus);
}

void GPUController::LoadLocal(void* pointer, int* errorStatus) {
    unsigned size = sizeElementY * sizeElementZ;
    if (sizeType == 8) {
        for (unsigned i = 0u; i < size; i++) {
            ((double*)pointer_Stream_CPU_input)[i] = ((double*)pointer)[i];
        }
    }
    else {
        *errorStatus = 1;
        return;
    }
    cublasStatus = cublasSetMatrixAsync(sizeElementY, sizeElementZ, sizeType, pointer_Stream_CPU_input, sizeElementX, pointer_Stream_GPU_input, sizeElementY, cudaStream);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus);
}

void GPUController::LaunchTask(int* errorStatus) {
    cublasStatus = cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)sizeElementX, (int)sizeElementZ, (int)sizeElementY,
        alpha, (const double*)pointer_GPU, (int)sizeElementX, (const double*)pointer_Stream_GPU_input, (int)sizeElementY,
        beta, (double*)pointer_Stream_GPU_result, (int)sizeElementX);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus)
}

void GPUController::UnloadLocal(void* pointer, int* errorStatus) {
    unsigned size = sizeElementX * sizeElementZ;
    cublasStatus = cublasGetMatrixAsync(sizeElementX, sizeElementZ, sizeType, pointer_Stream_GPU_result, sizeElementX, pointer_Stream_CPU_result, sizeElementX, cudaStream);
    RAISE_CUBLAS_ERROR(cublasStatus, errorStatus);
    cudaStatus = cudaStreamSynchronize(cudaStream);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    if (sizeType == 8) {
        for (unsigned i = 0u; i < size; i++) {
            ((double*)pointer)[i] = ((double*)pointer_Stream_CPU_result)[i];
        }
    }
    else {
        *errorStatus = 1;
        return;
    }
}

void GPUController::FreeGlobal(int* errorStatus) {
    cudaStatus = cudaFree(pointer_GPU);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaFreeHost(pointer_CPU);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
}

void GPUController::FreeLocal(int* errorStatus) {
    cudaStatus = cudaFree(pointer_Stream_GPU_result);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaFree(pointer_Stream_GPU_input);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaFreeHost(pointer_Stream_CPU_result);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
    cudaStatus = cudaFreeHost(pointer_Stream_CPU_input);
    RAISE_CUDA_ERROR(cudaStatus, errorStatus);
}

//Default Destructor for the GPU Controller class.
GPUController::~GPUController() {
    int errorStatus = 0;
    cublasStatus = cublasDestroy_v2(cublasHandle);
    RAISE_CUBLAS_ERROR(cublasStatus, &errorStatus);
    cudaStatus = cudaStreamDestroy(cudaStream);
    RAISE_CUDA_ERROR(cudaStatus, &errorStatus);
    delete alpha;
    delete beta;
}
