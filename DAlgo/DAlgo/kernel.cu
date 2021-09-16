
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <cmath>
#include <iostream>
using namespace std;

// Number of vertices in the graph
#define V 9
void create1DMap(int[], int, int*&, int*&);
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
void minDistance(int* dist, bool* sptSet, int *&min_index)
{

	// Initialize min value
	int min = INT_MAX;

	for (int v = 0; v < V; v++)
		if (sptSet[v] == false && dist[v] <= min)
			min = dist[v], *min_index = v;
}

__global__ void print2dArr(int *arr)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	printf("we are here");
	for (int x = 0; x < V; x++)
	{
		printf("%d", arr[i*V+x]);

		if (x + 1 % V == 0) printf("\n");
	}
}
__global__ void calcShortest(int *graph, int* dist, bool* sptSet, int minIndex)
{
	// We need to have each thread check a different dist location. If they all check the name location, then
	// We will have the highest thread < V update the value in dist arr

	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread < V)
	{
		int graphVal = graph[minIndex*V + thread];
		int distVal = dist[minIndex];
			if (!sptSet[thread] && graphVal && dist[minIndex] != INT_MAX
				&& ((dist[minIndex] + graphVal) < dist[thread]))
				dist[thread] = dist[minIndex]; + graphVal;
	}
}

// A utility function to print the constructed distance array
void printSolution(int* dist)
{
	for(int i = 0; i < V - 1; i++)
	{
		int getthis = sizeof(dist);
		printf("Vertex \t Distance from Source\n");
		printf(" \t\t%d\n", dist[i]);
	}
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(int *graph, int src)
{
	cudaError_t cudaError;

	size_t dSize = V * V * sizeof(int);
	// init an array of ptrs to more arrays
	int test2dArr[sizeof(dSize)];
	int *dev_graph;
	int* dist = new int[V]; // The output array.  dist[i] will hold the shortest
	int* dev_Dist;

	bool* sptSet = new bool[V]; // sptSet[i] will be true if vertex i is included in shortest

	// setting the max distance
	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX, sptSet[i] = false;

	dim3 threads = 32; // declaring the amount of threads based on the size of V (nodes per dimension)
	dim3 blocks = ((ceil(static_cast<float>(V) / static_cast<float>(32)))); // defining the number of blocks that will be required.
	// Initialize all distances as INFINITE and stpSet[] as false

	// Distance of source vertex from itself is always 0
	dist[src] = 0;

	// Find shortest path for all vertices
		int *u = new int(0);
		int minIndex = 0;
		// Pick the minimum distance vertex from the set of vertices not
		// yet processed. u is always equal to src in the first iteration.
		minDistance(dist, sptSet, u);
		minIndex = *u;
		// Mark the picked vertex as processed
		sptSet[minIndex] = true;

		// Alloc and then copy ptr to 
		cudaError = cudaMalloc((void**)&dev_Dist, V * sizeof(int));
		if (cudaError != cudaSuccess)
		{
			fprintf(stderr, "%s", cudaError);
		}
		cudaError = cudaMemcpy(dev_Dist, dist, V * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaError != cudaSuccess)
		{
			fprintf(stderr, "%s", cudaError);
		}
		cudaMalloc((void**)&dev_graph, dSize);
		cudaMemcpy(dev_graph, graph, dSize, cudaMemcpyHostToDevice);
		
		calcShortest<<<blocks, threads >>>(dev_graph, dev_Dist, sptSet, minIndex);
		cudaMemcpy(dist, dev_Dist, dSize, cudaMemcpyDeviceToHost);

		cudaFree(dev_Dist);

		printSolution(dist);
		// Update dist[v] only if is not in sptSet, there is an edge from
		// u to v, and total weight of path from src to  v through u is
		// smaller than current value of dist[v]
}

// driver program to test above function
int main()
{
	int* hostDist = { 0 };
	int* arrPitch = new int;
	/* Let us create the example graph discussed above */
	int graph[] = { 4, 0, 0, 0, 0, 0, 8, 0,
					8, 0, 0, 0, 0, 11, 0,
					7, 0, 4, 0, 0, 2,
					9, 14, 0, 0, 0,
					10, 0, 0, 0,
					2, 0, 0,
					1, 6,
					7}; // this is out simplified graph

	// We can convert the 2d array into a 1d array in qhich we construct a map. 
	// We can create an array of size ArrSize ==> [(1+n)n]/2 n is the width and 
	// // ceil(sqrt(ArrSize*2)) will provide us the width/height of our 2d array.
	// this will produce an array of size ceil(sqrt(n*2))^2 * sizeof(int)

	// ceil(sqrt(n*2)) will provide us the width/height of our 2d array.

	create1DMap(graph, sizeof(graph), hostDist, arrPitch);
	int getit = hostDist[7];
	dijkstra(hostDist, 0);
	
	cudaDeviceSynchronize();

	return 0;
}

void create1DMap(int inputArr[], int inputSize, int *&outputArr, int *&ouputPitch)
{
	int elements = inputSize/ sizeof(int);
	int arrPitch = ceil(sqrt(static_cast<double>(elements * 2)));
	*ouputPitch = arrPitch;
	int totalElements = pow(arrPitch, 2.0);
	outputArr = new int[totalElements];
	memset(outputArr, 0, totalElements * sizeof(int));
	int topArr = 0, botArr = 0;
	for (int i = 0; i < arrPitch - 1; i++)
	{
		for (int x = 0; x < arrPitch - (i + 1); x++)
		{
			outputArr[(i * (arrPitch + 1)) + x + 1] = inputArr[topArr];
			topArr++;
		}
		for (int y = 0; y < arrPitch - (i + 1); y++)
		{
			outputArr[((y + i + 1) * arrPitch) + i] = inputArr[botArr];
			botArr++;
		}
	}
	for (int i = 0; i < totalElements; i++)
	{
		if (i % arrPitch == 0)
		{
			std::cout << endl;
		}
		std::cout << outputArr[i] << " ";
	}
}