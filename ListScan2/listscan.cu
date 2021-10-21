// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

 
 
 //@@Fix the final output
__global__ void fixup(float *input, float *aux, int len) 
{
	unsigned int t = threadIdx.x, start = 2 * BLOCK_SIZE*blockIdx.x;

	if (blockIdx.x == 0)
	{
		if (t + start < len)
		{
			input[t + start] = aux[blockIdx.x - 1];
		}
		else
		{
			input[t] = 0;
		}

		if (t + start + BLOCK_SIZE < len)
		{
			input[t + start+ BLOCK_SIZE] = input[blockIdx.x - 1];
		}
		else
		{
			input[t + BLOCK_SIZE] = 0;
		}
	}
}  
  
  
  
__global__ void scan(float *input, float *output, float *aux,int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls
 
	__shared__ float scanArr[BLOCK_SIZE >> 1];

	unsigned int t = threadIdx.x, start = 2 * blockIdx.x*BLOCK_SIZE;
   
	if (t + start < len)
	{
		scanArr[t] = input[t + start];
    }
	else
	{
		scanArr[t] = 0;
	}

	if (t + start + BLOCK_SIZE < len)
	{
		scanArr[t + BLOCK_SIZE] = input[t + start + BLOCK_SIZE];
	}
	else
	{
		scanArr[t + BLOCK_SIZE] = 0;
	}
	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride <<= 1)
	{
		int index = (t + 1)*stride * 2 - 1;

		if (index < 2 * BLOCK_SIZE)
		{
			scanArr[index] += scanArr[index - stride];
		}
		__syncthreads();
    }

	for (stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1)
	{
		int index = (t + 1)*stride * 2 - 1;

		if (index + stride < 2 * BLOCK_SIZE)
		{
			scanArr[index + stride] += scanArr[index];
		}
		__syncthreads();
	}

	if (t + start < len)
		output[t + start] = scanArr[t];

	if (t + start + BLOCK_SIZE < len)
		output[t + start + BLOCK_SIZE] = scanArr[t + BLOCK_SIZE];

	if (t == 0 &&aux)
	{
		aux[blockIdx.x] = scanArr[(2 * BLOCK_SIZE) - 1];
	}
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray;
  float *deviceAuxScannedArray;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  //@@Allocating GPU Memory		
  wbTime_start(GPU, "Allocating GPU memory.");
 
  cudaMalloc((void**)&deviceInput, sizeof(float)*numElements);
  cudaMalloc((void**)&deviceOutput, sizeof(float)*numElements);

  cudaMalloc((void**)&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(float));
  cudaMalloc((void**)&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(float));
    
 
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@Clear output memory using cudaMemset
  wbTime_start(GPU, "Clearing output memory.");
  
  cudaMemset(deviceOutput, 0, sizeof(float)*numElements);
  
  wbTime_stop(GPU, "Clearing output memory.");

  
  //@@Copying input memory to the GPU
  wbTime_start(GPU, "Copying input memory to the GPU.");
  
  wbCheck(cudaMemcpy(deviceInput, hostInput, sizeof(float)*numElements, cudaMemcpyHostToDevice));
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  
  
  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil((float)numElements / (BLOCK_SIZE << 1));
   
  dim3 dimGrid(numBlocks);
  dim3 blockDim(BLOCK_SIZE);

  
  //@@Performing CUDA computation
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan << <dimGrid, blockDim >> > (deviceInput, deviceOutput, deviceAuxArray, numElements);
  
  cudaDeviceSynchronize();

  scan << <dimGrid, blockDim >> > (deviceAuxArray, deviceAuxScannedArray, NULL, (BLOCK_SIZE<<1));
 
  cudaDeviceSynchronize();

  fixup << <1, blockDim >> > (deviceOutput, deviceAuxScannedArray, numElements);
  
  wbTime_stop(Compute, "Performing CUDA computation");
  
  
  //@@Copying output memory to the CPU
  wbTime_start(Copy, "Copying output memory to the CPU");
  
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float)*numElements, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  
  
  //@@Freeing GPU Memory
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);
  
  
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  
  //@@Verification of the result
  wbSolution(args, hostOutput, numElements);

  
  //@@Freeing the host memory
  free(hostInput);
  free(hostOutput);

  return 0;
}
