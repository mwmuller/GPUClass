#include <wb.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 12 // output tile width
#define w (TILE_WIDTH + Mask_width - 1) // input tile width
#define clamp(x) (min(max((x), 0.0), 1.0))
#define Channels 3

//@@ INSERT CODE HERE
__constant__ float mask[Mask_width * Mask_width];

__global__ void convolution(float *deviceInputImageData, const float * __restrict__ deviceMaskData,
	float *deviceOutputImageData, int imageChannels,
	int imageWidth, int imageHeight)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	int row_o = ty + TILE_WIDTH * blockIdx.y;
	int col_o = tx + TILE_WIDTH * blockIdx.x;

	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;

	__shared__ float s_input[w][w]; // input shared memory. 

	if (tx < imageWidth && ty < imageHeight)
	{
		for (int c = 0; c < Channels; c++)
		{
			if ((row_i >= 0) && (row_i < imageHeight) &&
				(col_i >= 0) && (col_i < imageWidth))
			{
				s_input[ty][tx] = deviceInputImageData[(row_i * imageWidth + col_i) * Channels + c];
			}
			else
			{
				s_input[ty][tx] = 0.0f;
			}

			__syncthreads();

			float output = 0.0f;
			if (tx < TILE_WIDTH && ty < TILE_WIDTH)
			{
				for (int i = 0; i < Mask_width; i++)
				{
					for (int j = 0; j < Mask_width; j++)
					{
						output += s_input[i + ty][(j + tx)] * deviceMaskData[(i * Mask_width + j)];
					}
				}
			}
			__syncthreads();
			if (row_o < imageHeight && col_o < imageWidth)
			{
				deviceOutputImageData[(row_o * imageWidth + col_o) * Channels + c] = clamp(output);
			}
		}
		__syncthreads();
	}
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE

  cudaMalloc((void**)&deviceInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
  cudaMalloc((void**)&deviceOutputImageData, imageHeight * imageWidth * imageChannels * sizeof(float));
  cudaMalloc((void**)&deviceMaskData, 25 * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageHeight * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputImageData, hostOutputImageData, imageHeight * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, 25 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, hostMaskData, Mask_width * Mask_width * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid(w, w);
  dim3 dimBlock(ceil((float)imageWidth /TILE_WIDTH), ceil((float)imageHeight/TILE_WIDTH));

  convolution<<<dimBlock, dimGrid>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);

  printf("Host data %f", hostOutputImageData[0]);
  printf("Host data %f", hostOutputImageData[1]);
  printf("Host data %f", hostOutputImageData[2]);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(mask);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
