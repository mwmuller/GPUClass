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

#define TILE_WIDTH 16
#define MULTI_TILE 4

__global__ void matrixMultiplyTiled(const float * A, const float * B, float * C, int numARows,
	int numAColumns, int numBColumns) {
	// TODO: implement this function
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = ty + blockDim.y * by; // x index of our thread in the block grid
	int col = tx + blockDim.x * bx; // y index of the thread
	float Cvalue = 0;

	for (int i = 0; i < ceil((float)numAColumns / TILE_WIDTH); i++)
	{

		if ((i*TILE_WIDTH + tx) < numAColumns && row < numARows)
		{
			ds_A[ty][tx] = A[row*numAColumns + (i * TILE_WIDTH + tx)];
		}
		else
		{
			ds_A[ty][tx] = 0;
		}

		if ((i*TILE_WIDTH + ty) < numAColumns && col < numBColumns)
		{
			ds_B[ty][tx] = B[(i*TILE_WIDTH + ty)*numBColumns + col];
		}
		else
		{
			ds_B[ty][tx] = 0;
		}

		__syncthreads();

		for (int p = 0; p < TILE_WIDTH; p++)
		{
			Cvalue += ds_A[ty][p] * ds_B[p][tx];
		}
		__syncthreads();
	}
	if (row < numARows && col < numBColumns)
	{
		C[row*numBColumns + col] = Cvalue;
	}
}

__global__ void matrixMultiplyMultiTile(const float * A, const float * B, float * C, int numARows,
	int numAColumns, int numBColumns) {

	// TODO: implement this function
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH*MULTI_TILE];
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = ty + blockDim.y * by;
	int col = tx + blockDim.x * bx;
	float Cvalue[MULTI_TILE] = { 0 };

	int tileMultipleCol = col + (blockIdx.x * TILE_WIDTH * (MULTI_TILE - 1));

	if (tileMultipleCol < numBColumns)
	{
		for (int i = 0; i < ceil((float)numAColumns) / (TILE_WIDTH); i++)
		{

			if ((i*TILE_WIDTH + tx) < numAColumns && row < numARows)
			{
				ds_A[ty][tx] = A[row*numAColumns + (i * TILE_WIDTH + tx)];
			}
			else
			{
				ds_A[ty][tx] = 0;
			}

			for (int f = 0; f < MULTI_TILE; f++) // Include the next Y block of B 
			{
				if ((i*TILE_WIDTH + ty) < numAColumns && (tileMultipleCol + (f * TILE_WIDTH) < numBColumns))
				{
					ds_B[ty][tx + (f * TILE_WIDTH)] = B[(i * TILE_WIDTH + ty)*numBColumns + (tileMultipleCol)+(f * TILE_WIDTH)];
				}
				else
				{
					ds_B[ty][tx + (f * TILE_WIDTH)] = 0;
				}
			}
			__syncthreads();
			for (int f = 0; f < MULTI_TILE; f++) // Include the next Y block of B 
			{
				for (int p = 0; p < (TILE_WIDTH); p++)
				{
					Cvalue[f] += ds_A[ty][p] * ds_B[p][tx + (f * TILE_WIDTH)];
				}
			}
			__syncthreads();
		}
		for (int f = 0; f < MULTI_TILE; f++)
		{
			if (row < numARows && (tileMultipleCol + (f * TILE_WIDTH)) < numBColumns)
			{
				C[row*numBColumns + (tileMultipleCol + (f * TILE_WIDTH))] = Cvalue[f]; // index 16, row1, col 0, m 0
			}
		}
	}
}


void matrixMultiplyHost(const float * A, const float * B, float * C, int numARows,
	int numAColumns, int numBColumns) {

	for (int i = 0; i < numARows; ++i) {

		for (int j = 0; j < numBColumns; ++j) {

			float value = 0;

			for (int k = 0; k < numAColumns; ++k) {

				value += A[i * numAColumns + k] * B[k * numBColumns + j];

			}

			C[i * numBColumns + j] = value;

		}

	}

}

int main(int argc, char **argv) {

	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
		&numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
		&numBColumns);

	//Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;

	//Allocate the hostC matrix
	hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

	wbTime_start(GPU, "Allocating GPU memory.");

	// TODO: allocate GPU memory
	cudaMalloc((void**)&deviceA, sizeof(float)*numARows*numAColumns);
	cudaMalloc((void**)&deviceB, sizeof(float)*numAColumns*numBColumns);
	cudaMalloc((void**)&deviceC, sizeof(float)*numARows*numBColumns);
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");

	// TODO: copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, sizeof(float)*numAColumns*numBColumns, cudaMemcpyHostToDevice);
	cudaMemset(deviceC, 0, sizeof(float)*numARows*numBColumns);

	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// TODO: initialize the grid and block dimensions
	dim3 dimGrid(TILE_WIDTH, TILE_WIDTH);
	dim3 dimBlock(ceil(((float)numBColumns) / dimGrid.x), ceil(((float)numARows) / dimGrid.y));

	wbTime_start(Compute, "Performing basic tiled computation");

	// TODO: Launch the basic tiled GPU Kernel here
	matrixMultiplyTiled << <dimBlock, dimGrid >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing basic tiled computation");


	wbTime_start(Copy, "Copying output memory to the CPU");

	// TODO: copy the GPU memory back to the CPU here

	cudaMemcpy(hostC, deviceC, sizeof(float)*numARows*numBColumns, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	wbTime_stop(Copy, "Copying output memory to the CPU");

	// check the basic tiled solution
	wbSolution(args, hostC, numCRows, numCColumns);

	memset(hostC, 0, sizeof(float)*numARows*numBColumns);
	cudaFree(deviceC);
	cudaMalloc((void**)&deviceC, sizeof(float)*numARows*numBColumns);
	cudaMemset(deviceC, 0, sizeof(float)*numARows*numBColumns);

	wbTime_start(Compute, "Performing multi-tiled computation");

	matrixMultiplyMultiTile << <dimBlock, dimGrid >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
	cudaDeviceSynchronize();

	wbTime_stop(Compute, "Performing multi-tiled computation");

	wbTime_start(Copy, "Copying output memory to the CPU 2");

	cudaMemcpy(hostC, deviceC, sizeof(float)*numARows*numBColumns, cudaMemcpyDeviceToHost);
	// TODO: copy the GPU memory back to the CPU here
	for (int i = 0; i < numARows*numBColumns; i++)
	{
		printf("%f ,", hostC[i]);
	}
	wbTime_stop(Copy, "Copying output memory to the CPU 2");

	// check the multi-tiled solution
	wbSolution(args, hostC, numCRows, numCColumns);


	wbTime_start(GPU, "Freeing GPU Memory");

	// TODO: Free the GPU memory here
	cudaFree(deviceC);
	cudaFree(deviceA);
	cudaFree(deviceB);
	wbTime_stop(GPU, "Freeing GPU Memory");

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}
