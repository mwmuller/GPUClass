
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define threadSIZE  1024 // max thread count per cuda
#define blockSize   16 // max number of blocks
#define arrScale 75 // amount of times to multiple threads
#define arrSize  (threadSIZE * arrScale)

cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size);

__global__ void addKernel(int *c, int *a, int *b)
{
    int j = threadIdx.x;
    c[j] = a[j] + b[j];

    for (int i = 1; i < arrScale; i++)
    {
        int index = (threadSIZE * i) + j;

        c[index] = a[index] + b[index];
    }
}

__global__ void printArr(int* arr)
{
    for (int i = 0; i < arrSize; i++)
    {
        printf("%d, ", arr[i]);
    }
}

__global__ void fillArrs(int* aArray, int *bArray)
{
    int j = threadIdx.x;
    
    aArray[j] = j + 1;
    bArray[j] = j * 2;
    // move backwards from the max

    for (int i = 1; i < arrScale; i++)
    {
        int index = (threadSIZE * i) + j;
        aArray[index] = index + 1; // 1024 * the scalar + the initial index
        bArray[index] = index * 2; // 1024 * the scalar + the initial index
    }

}

int main()
{

    cudaError_t cudaStatus;
    // an array of 4 points;
    int a[arrSize] = { 0 };
    int b[arrSize] = { 0 };
    int c[arrSize] = { 0 };

    size_t sizeCount = arrSize * sizeof(int);

    int* dev_a = new int[arrSize];
    int* dev_b = new int[arrSize];
    int* dev_c = new int[arrSize];

    cudaStatus = cudaMalloc((void**)&dev_a, sizeCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy awdawd failed!");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, sizeCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy awdawd failed!");
        return 1;
    }
    cudaStatus = cudaMemcpy(dev_a, a, sizeCount, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy awdawd failed!");
        return 1;
    }
    cudaStatus = cudaMemcpy(dev_b, b, sizeCount, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy awdawd failed!");
        return 1;
    }

    // filling array with numbers
    fillArrs <<<1, threadSIZE>>>(dev_a, dev_b);

    cudaStatus = cudaMemcpy(a, dev_a, arrSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(b, dev_b, arrSize * sizeof(int), cudaMemcpyDeviceToHost);
    // Add vectors in parallel.

    cudaStatus = addWithCuda(c, a, b, arrSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, arrSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(dev_c, c, arrSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }
    
    printf("Array C {");
    printArr<<<1, 1>>>(dev_c);
    printf("} \n");
    

    printf("done");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size)
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

    // Allocate GPU buffers for three vectors (two input, one output).
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
    addKernel<<<1, threadSIZE>>>(dev_c, dev_a, dev_b);

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
