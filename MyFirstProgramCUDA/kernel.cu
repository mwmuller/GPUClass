
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

// Number of vertices in the graph
#define V 9


typedef int my_arr[V];
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
void minDistance(int dist[], bool sptSet[], int* &min_index)
{

    // Initialize min value
    int min = INT_MAX;

    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = new int(v);
}

__global__ void print2dArr(my_arr *arr)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		for (int x = 0; x < V; x++)
		{
			printf("%d", arr[i][x]);
			if (x + 1 % V == 0) printf("\n");
		}
}
__global__ void calcShortest(my_arr *graph, int* dist, bool* sptSet, int minIndex)
{
	// We need to have each thread check a different dist location. If they all check the name location, then
	// We will have the highest thread < V update the value in dist arr
	int thread = blockIdx.x*blockDim.x + threadIdx.x;
	if (thread < V)
	{
		if (!sptSet[thread] && graph[minIndex][thread] && dist[minIndex] != INT_MAX
			&& ((dist[minIndex] + graph[minIndex][thread]) < dist[thread]))
			dist[thread] = dist[minIndex] + graph[minIndex][thread];
	}
}

// A utility function to print the constructed distance array
__global__ void printSolution(int* dist)
{
    printf("Vertex \t Distance from Source\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        printf(" \t\t%d\n", dist[i]);
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
int* dijkstra(my_arr *graph, int src)
{
	size_t dSize = V * V * sizeof(int);
    // init an array of ptrs to more arrays
	my_arr *dev_graph;

    int* dist = { 0 }; // The output array.  dist[i] will hold the shortest
	int* dev_Dist = { 0 };

    bool* sptSet = new bool[V]; // sptSet[i] will be true if vertex i is included in shortest

	// setting the max distance
	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX, sptSet[i] = false;

	dim3 threads = 1024; // declaring the amount of threads based on the size of V (nodes per dimension)
	dim3 blocks = ((ceil(static_cast<float>(V) / static_cast<float>(1024)))); // defining the number of blocks that will be required.
	// Initialize all distances as INFINITE and stpSet[] as false

	// Distance of source vertex from itself is always 0
	dist[src] = 0;

	// Find shortest path for all vertices
	for (int count = 0; count < V - 1; count++) {
		int *u = 0;
		int minIndex = 0;
		// Pick the minimum distance vertex from the set of vertices not
		// yet processed. u is always equal to src in the first iteration.
		minDistance(dist, sptSet, u);
		minIndex = *u;
		// Mark the picked vertex as processed
		sptSet[minIndex] = true;

		// Alloc and then copy ptr to 
		cudaMalloc((void**)&dev_Dist, V * sizeof(int));
		cudaMemcpy(dev_Dist, dist, V * sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_graph, dSize);
		cudaMemcpy(dev_graph, graph, dSize, cudaMemcpyHostToDevice);

		calcShortest<<<blocks, threads>>> (dev_graph, dev_Dist, sptSet, minIndex);

		cudaDeviceSynchronize();
		cudaMemcpy(dist, dev_Dist, dSize, cudaMemcpyDeviceToHost);
		// Update dist[v] only if is not in sptSet, there is an edge from
		// u to v, and total weight of path from src to  v through u is
		// smaller than current value of dist[v]
		
	}



    return dist;
}

// driver program to test above function
int main()
{
    int* hostDist = new int[V];
	size_t dSize = V * V * sizeof(int);
	my_arr *dArr;
	dArr = (my_arr *)malloc(dSize);
	memset(dArr, 0, dSize);
    /* Let us create the example graph discussed above */
    int graph[V][V] = { { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
                        { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
                        { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
                        { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
                        { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
                        { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
                        { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
                        { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
                        { 0, 0, 2, 0, 0, 0, 6, 7, 0 } };
	memcpy(dArr, graph,dSize);
	hostDist = dijkstra(dArr, 0);

    printSolution<<<1, V>>>(hostDist);

    return 0;
}

// This code is contributed by shivanisinghss2110