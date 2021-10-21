#include <wb.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define NUM_BINS 4096
#define SATURATION 127
#define THREADS 256

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

//@@Add the Kernel Code here
/// Will construct the bins
__global__ void histogram(unsigned int *input, unsigned int *bins, unsigned int inputLength)
{
	__shared__ unsigned int bins_s[NUM_BINS];
	__shared__ unsigned int temp;
	unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x; // get me 0 to 4095

	if (thread < NUM_BINS) // psuedo memset to 0
	{
		bins_s[thread] = 0;
	}
	__syncthreads();
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int threadIndex = thread;

	while (threadIndex < inputLength)
	{
		unsigned int binIndex = input[threadIndex];

		if (bins_s[binIndex] < 127)
		{
			atomicAdd(&bins_s[binIndex], 1);
		}
		threadIndex += stride;
	}
	__syncthreads();

	if (thread < inputLength)
	{
		atomicAdd(&bins[thread], bins_s[thread]);
	}
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins; // output array

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength); 
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc((void**)&deviceBins, sizeof(unsigned int) * NUM_BINS);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int)*inputLength, cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, sizeof(unsigned int)*NUM_BINS);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");

  dim3 gridDim(ceil(((float)inputLength / THREADS)));
  dim3 blockDim(THREADS);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  histogram << <gridDim, blockDim >> > (deviceInput, deviceBins, inputLength);


  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int)*NUM_BINS, cudaMemcpyDeviceToHost);
  for(int i = 0; i < NUM_BINS; i++)
  {
	  if (i == 1098)
	  {
		  printf("index %d = %d \n", i, hostBins[i]);
	  }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  cudaFree(deviceBins);
  cudaFree(deviceInput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
