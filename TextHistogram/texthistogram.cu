#include <wb.h>

#define NUM_BINS 128

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

__global__ void histogram_kernel(unsigned char *input,unsigned int *bins,unsigned int numElements)
{
      __shared__ unsigned int bins_s[NUM_BINS];
      
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(threadIdx.x<NUM_BINS)
            bins_s[threadIdx.x] = 0;
            
      __syncthreads(); 
      
      
      unsigned int stride = blockDim.x * gridDim.x;
      
      while(i<numElements)
      {
         atomicAdd(&bins_s[(unsigned int)input[i]],1);
         
         i+=stride;
      
      } 
      
      __syncthreads(); 
      
      
      if(threadIdx.x<NUM_BINS)
          atomicAdd(&bins[threadIdx.x],bins_s[threadIdx.x]);    



}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned char *hostInput;
  unsigned int *hostBins;
  unsigned char *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned char *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  
  CUDA_CHECK(cudaMalloc((void**)&deviceInput,inputLength));
  CUDA_CHECK(cudaMalloc((void**)&deviceBins,NUM_BINS * sizeof(unsigned int)));
 
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput,hostInput,inputLength,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(deviceBins,0,NUM_BINS * sizeof(unsigned int)));
  
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  
  dim3 grid(30);
  dim3 block(256);
  
  histogram_kernel<<<grid,block>>>(deviceInput,deviceBins,inputLength);
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins,deviceBins,NUM_BINS * sizeof(unsigned int),cudaMemcpyDeviceToHost));
  
  
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
