
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <cmath>
#include <iostream>
using namespace std;


// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
void minDistance(int* dist, bool* sptSet, int *min_index, int arrPitch)
{

	// Initialize min value
	int min = INT_MAX;

	for (int v = 0; v < arrPitch; v++)
		if (sptSet[v] == false && dist[v] <= min)
			min = dist[v], *min_index = v;
}

__global__ void calcShortest(int *graph, int* dist, bool* sptSet, int minIndex, int arrPitch)
{
	// We need to have each thread check a different dist location. If they all check the name location, then
	// We will have the highest thread < V update the value in dist arr

	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread < arrPitch)
	{
		int graphVal = graph[minIndex*arrPitch + thread];
		int distVal = dist[minIndex];
			if (!sptSet[thread] && graphVal && distVal != INT_MAX
				&& ((distVal + graphVal) < dist[thread]))
				dist[thread] = distVal + graphVal;
	}
}

// A utility function to print the constructed distance array
void printSolution(int* dist, int arrPitch)
{
	for(int i = 0; i < arrPitch - 1; i++)
	{
		printf("Vertex \t Distance from Source\n");
		printf(" \t\t%d\n", dist[i]);
	}
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(int *graph, int src, int* arrPitch)
{
	cudaError_t cudaError;
	int thisPitch = *arrPitch;
	size_t dSize = thisPitch * thisPitch * sizeof(int);
	// init an array of ptrs to more arrays for device
	int *dev_graph = 0;
	int* dist = new int[thisPitch]; // The output array.  dist[i] will hold the shortest
	int* dev_Dist = 0;
	bool* devicesptSet = false;
	bool* sptSet = new bool[thisPitch]; // sptSet[i] will be true if vertex i is included in shortest

	// setting the max distance
	for (int i = 0; i < thisPitch; i++)
		dist[i] = INT_MAX, sptSet[i] = false;

	dim3 threads = 32; // declaring the amount of threads based on the size of V (nodes per dimension)
	dim3 blocks = ((ceil(static_cast<float>(thisPitch) / static_cast<float>(32)))); // defining the number of blocks that will be required.
	// Initialize all distances as INFINITE and stpSet[] as false

	// Distance of source vertex from itself is always 0
	dist[src] = 0;
	cudaMalloc((void**)&devicesptSet, thisPitch*sizeof(bool));
	cudaMalloc((void**)&dev_graph, dSize);
	cudaMemcpy(dev_graph, graph, dSize, cudaMemcpyHostToDevice);
	for (int count = 0; count < thisPitch - 1; count++)
	{
		// Find shortest path for all vertices
		int *u = new int(0);
		int minIndex = 0;
		// Pick the minimum distance vertex from the set of vertices not
		// yet processed. u is always equal to src in the first iteration.
		minDistance(dist, sptSet, u, thisPitch);
		minIndex = *u;
		// Mark the picked vertex as processed
		sptSet[minIndex] = true;

		cudaMemcpy(devicesptSet, sptSet, dSize, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_graph, dSize);
		cudaMemcpy(dev_graph, graph, dSize, cudaMemcpyHostToDevice);
		// Alloc and then copy ptr to 
		cudaError = cudaMalloc((void**)&dev_Dist, thisPitch * sizeof(int));
		if (cudaError != cudaSuccess)
		{
			fprintf(stderr, "%s", cudaError);
		}
		cudaError = cudaMemcpy(dev_Dist, dist, thisPitch * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaError != cudaSuccess)
		{
			fprintf(stderr, "%s", cudaError);
		}

		calcShortest << <blocks, threads >> > (dev_graph, dev_Dist, sptSet, minIndex, thisPitch);
		cudaMemcpy(dist, dev_Dist, dSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(sptSet, devicesptSet, dSize, cudaMemcpyDeviceToHost);
	}

		cudaFree(dev_Dist);

		printSolution(dist, thisPitch);
		// Update dist[v] only if is not in sptSet, there is an edge from
		// u to v, and total weight of path from src to  v through u is
		// smaller than current value of dist[v]
}

__global__ void create1DMap(int* inputArr, int *outputArr, int *inPitch)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arrPitch = *inPitch;
	int inputElements = (arrPitch*(arrPitch - 1)) / 2;


	int topArr = (((i + 2) * (i + 1)) / 2) - 1;
	int botArr = topArr;
	if (i < arrPitch - 1)
	{
		for (int x = 0; x < arrPitch - (i + 1); x++)
		{
			int getit = inputArr[topArr];
			int index = (i * (arrPitch + 1)) + x + 1;
			outputArr[index] = inputArr[topArr];
			topArr++;
		}
		for (int y = 0; y < arrPitch - (i + 1); y++)
		{
			int getity = inputArr[botArr];
			int indexy = ((y + i + 1) * arrPitch) + i;
			outputArr[indexy] = inputArr[botArr];
			botArr++;
		}
	}
}

// driver program to test above function
int main()
{
	int* hostGraph;
	int* arrPitch = new int(0);

	// Device values needed
	int* devInGraph;
	int* devOutGraph = 0;
	int* devDist;
	int* devInArrSize;
	int* devArrPitch;
	/* Let us create the example graph discussed above */
	
	int graph[] = { 4, 0, 0, 0, 0, 0, 8, 0,
					8, 0, 0, 0, 0, 11, 0,
					7, 0, 4, 0, 0, 2,
					9, 14, 0, 0, 0,
					10, 0, 0, 0,
					2, 0, 0,
					1, 6,
					7}; // this is out simplified graph
					
	//int graph[] = { 1,0,2,5,4,6 };

	// We can create an array of size ArrSize ==> [(1+n)n]/2 n is the width and 
	// ceil(sqrt(ArrSize*2)) will provide us the width/height of our 2d array.
	// this will produce an array of size ceil(sqrt(n*2))^2 * sizeof(int)
	int elements = sizeof(graph) / sizeof(int);

	// ceil(sqrt(ArrSize*2)) will provide us the width/height of our 2d array.
	*arrPitch = ceil(sqrt(static_cast<double>(elements * 2)));

	// Declare the max cuda size needed
	int totalCudaMallocSize = (*arrPitch) * (*arrPitch) * sizeof(int);


	// ceil(sqrt(n*2)) will provide us the width/height of our 2d array.
	dim3 threads = 32; // declaring the amount of threads based on the size of V (nodes per dimension)
	dim3 blocks = ((ceil(static_cast<float>(*arrPitch) / static_cast<float>(32)))); // defining the number of blocks that will be required.

	hostGraph = (int*)malloc(totalCudaMallocSize);
	memset(hostGraph, 0, totalCudaMallocSize);

	cudaMalloc((void**)&devInGraph, elements * sizeof(int));
	cudaMemcpy(devInGraph, graph, elements * sizeof(int), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&devOutGraph, totalCudaMallocSize);
	cudaMemset(devOutGraph, 0, totalCudaMallocSize);

	cudaMalloc((void**)&devArrPitch, sizeof(int));
	cudaMemcpy(devArrPitch, arrPitch, sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&devInArrSize, sizeof(int));
	cudaMemcpy(devArrPitch, arrPitch, sizeof(int), cudaMemcpyHostToDevice);

	create1DMap<<<blocks, threads>>>(devInGraph, devOutGraph, devArrPitch);

	cudaDeviceSynchronize();
	int hostSize = (*arrPitch)* (*arrPitch);
	cudaMemcpy(hostGraph, devOutGraph, (*arrPitch)* (*arrPitch) * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < hostSize; i++)
	{
		int getit = hostGraph[i];
		if (i % *arrPitch == 0)
		{
			printf("\n");
		}
		printf("%d  ", getit);
	}
	dijkstra(hostGraph, 0, arrPitch);
	
	cudaDeviceSynchronize();

	return 0;
}
