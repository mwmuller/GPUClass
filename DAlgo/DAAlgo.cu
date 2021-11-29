#include <sstream>
#include <vector>
#include <iostream>
#include <time.h>
#include <float.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_ASYNCHRONOUS_ITERATIONS 3  // Number of async loop iterations before attempting to read results back
#define ASYNC_THRESHOLD 2000 // Used to determine if more iterations are required. (Max of 25)
#define ASYNC_THRESHOLD_NEIGHBORS 25 // For each 10 neighbors add another iteration
#define BLOCK_SIZE 32


// --- The graph data structure is an adjacency list.
typedef struct {

	// --- Contains the integer offset to point to the edge list for each vertex
	int *vertexArray;

	// --- Overall number of vertices
	int numVertices;

	// --- Contains the "destination" vertices each edge is attached to
	int *edgeArray;

	// --- Overall number of edges
	int numEdges;

	// --- Contains the weight of each edge
	unsigned int *weightArray;

} GraphData;

/**********************************/
/* GENERATE RANDOM GRAPH FUNCTION */
/**********************************/
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex) {

	graph->numVertices = numVertices;
	graph->vertexArray = (int *)malloc(graph->numVertices * sizeof(int));
	graph->numEdges = numVertices * neighborsPerVertex;
	graph->edgeArray = (int *)malloc(graph->numEdges * sizeof(int));
	graph->weightArray = (unsigned int *)malloc(graph->numEdges * sizeof(unsigned int));

	for (int i = 0; i < graph->numVertices; i++) graph->vertexArray[i] = i * neighborsPerVertex;

	int *tempArray = (int *)malloc(neighborsPerVertex * sizeof(int));
	for (int k = 0; k < numVertices; k++) {
		for (int l = 0; l < neighborsPerVertex; l++) tempArray[l] = INT_MAX;
		for (int l = 0; l < neighborsPerVertex; l++) {
			bool goOn = false;
			int temp;
			while (goOn == false) {
				goOn = true;
				temp = (rand() % graph->numVertices); // move to 0;
				for (int t = 0; t < neighborsPerVertex; t++)
					if (temp == tempArray[t]) goOn = false;
				if (temp == k) goOn = false;
				if (goOn == true) tempArray[l] = temp;
			}
			graph->edgeArray[k * neighborsPerVertex + l] = temp;
			graph->weightArray[k * neighborsPerVertex + l] = (rand() % 1000 + 1);
		}
	}
}

/************************/
/* minDistance FUNCTION */
/************************/
// --- Finds the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree
int minDistance(unsigned int *shortestDistances, bool *finalizedVertices, const int sourceVertex, const int N) {

	// --- Initialize minimum value
	int minIndex = sourceVertex;
	int min = INT_MAX;

	for (int v = 0; v < N; v++)
		if (finalizedVertices[v] == false && shortestDistances[v] <= min) min = shortestDistances[v], minIndex = v;

	return minIndex;
}

/************************/
/* dijkstraCPU FUNCTION */ // This will remain unchanged
/************************/
void dijkstraCPU(unsigned int *graph, unsigned int *h_shortestDistances, int sourceVertex, const int N) {

	// --- h_finalizedVertices[i] is true if vertex i is included in the shortest path tree
	//     or the shortest distance from the source node to i is finalized
	bool *h_finalizedVertices = (bool *)malloc(N * sizeof(bool));

	// --- Initialize h_shortestDistancesances as infinite and h_shortestDistances as false
	for (int i = 0; i < N; i++) h_shortestDistances[i] = INT_MAX, h_finalizedVertices[i] = false;

	// --- h_shortestDistancesance of the source vertex from itself is always 0
	h_shortestDistances[sourceVertex] = 0;

	// --- Dijkstra iterations
	for (int iterCount = 0; iterCount < N - 1; iterCount++) {

		// --- Selecting the minimum distance vertex from the set of vertices not yet
		//     processed. currentVertex is always equal to sourceVertex in the first iteration.
		int currentVertex = minDistance(h_shortestDistances, h_finalizedVertices, sourceVertex, N);

		// --- Mark the current vertex as processed
		h_finalizedVertices[currentVertex] = true;

		// --- Relaxation loop
		for (int v = 0; v < N; v++) {

			// --- Update dist[v] only if it is not in h_finalizedVertices, there is an edge
			//     from u to v, and the cost of the path from the source vertex to v through
			//     currentVertex is smaller than the current value of h_shortestDistances[v]
			if (!h_finalizedVertices[v] &&
				graph[currentVertex * N + v] &&
				h_shortestDistances[currentVertex] != INT_MAX &&
				h_shortestDistances[currentVertex] + graph[currentVertex * N + v] < h_shortestDistances[v])

				h_shortestDistances[v] = h_shortestDistances[currentVertex] + graph[currentVertex * N + v];
		}
	}
}

/***************************/
/* MASKARRAYEMPTY FUNCTION */
/***************************/
// --- Check whether all the vertices have been finalized. This tells the algorithm whether it needs to continue running or not.
bool allFinalizedVertices(bool *finalizedVertices, int numVertices) {

	for (int i = 0; i < numVertices; i++)
	{
		if (finalizedVertices[i] == true)
		{
			return false;
		}
	}

	return true;
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, unsigned int* __restrict__ d_shortestDistances, unsigned int* __restrict__ d_updatingShortestDistances,
	const int sourceVertex, const int numVertices) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numVertices) {

		if (sourceVertex == tid) {

			d_finalizedVertices[tid] = true;
			d_shortestDistances[tid] = 0;
			d_updatingShortestDistances[tid] = 0;
		}

		else {

			d_finalizedVertices[tid] = false;
			d_shortestDistances[tid] = INT_MAX;
			d_updatingShortestDistances[tid] = INT_MAX;
		}
	}
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel1(const int * __restrict__ vertexArray, const int* __restrict__ edgeArray,
	const unsigned int * __restrict__ weightArray, bool * __restrict__ finalizedVertices, unsigned int * __restrict__ shortestDistances,
	unsigned int * __restrict__ updatingShortestDistances, const int numVertices, const int numEdges) {
	int tx = threadIdx.x;
	int tid = blockIdx.x*blockDim.x + tx;
	__shared__ unsigned int s_shortest[BLOCK_SIZE]; 

	if (tid < numVertices) {
		s_shortest[tx] = shortestDistances[tid]; 

		__syncthreads();

		if (finalizedVertices[tid] == true) {
		
			finalizedVertices[tid] = false;

			int edgeStart = vertexArray[tid], edgeEnd; // get the edge index that we start at

			// Check if we are beyond the number of verticies that we can check
			if (tid + 1 < (numVertices)) edgeEnd = vertexArray[tid + 1]; // Check if we are in bounds. 
			else                         edgeEnd = numEdges; // We are at the max.

			for (int edge = edgeStart; edge < edgeEnd; edge++) {
				int nid = edgeArray[edge]; // get the ID which will be associated with a vertex
				atomicMin(&updatingShortestDistances[nid], s_shortest[tx] + weightArray[edge]); // assigns minimum value to uSD pointer
			}
		}
	}
}

/**************************/
/* DIJKSTRA GPU KERNEL #2 */
/**************************/
__global__  void Kernel2(const int * __restrict__ vertexArray, const int * __restrict__ edgeArray, const unsigned int* __restrict__ weightArray,
	bool * __restrict__ finalizedVertices, unsigned int* __restrict__ shortestDistances, unsigned int* __restrict__ updatingShortestDistances,
	const int numVertices) {
	int tx = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + tx;

	if (tid < numVertices) {

		if (shortestDistances[tid] > updatingShortestDistances[tid]) {
			shortestDistances[tid] = updatingShortestDistances[tid];
			finalizedVertices[tid] = true;
		}

		//__syncthreads();
		updatingShortestDistances[tid] = shortestDistances[tid];
	}
}

/************************/
/* dijkstraGPU FUNCTION */
/************************/
void dijkstraGPU(GraphData *graph, const int sourceVertex, unsigned int * __restrict__ h_shortestDistances) {

	//Init of GPU timing 
	float elapsed = 0, elapsedComp = 0;
	cudaEvent_t start1, stop1, start0, stop0;

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEventRecord(start1, 0);

	// END of init/Start

	// --- Create device-side adjacency-list, namely, vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
	int     *d_vertexArray;         cudaMalloc(&d_vertexArray, sizeof(int)   * graph->numVertices);
	int     *d_edgeArray;           cudaMalloc(&d_edgeArray, sizeof(int)   * graph->numEdges);
	unsigned int   *d_weightArray;         cudaMalloc(&d_weightArray, sizeof(unsigned int) * graph->numEdges);

	// --- Copy adjacency-list to the device
	cudaMemcpy(d_vertexArray, graph->vertexArray, sizeof(int)   * graph->numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgeArray, graph->edgeArray, sizeof(int)   * graph->numEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weightArray, graph->weightArray, sizeof(int) * graph->numEdges, cudaMemcpyHostToDevice);

	// --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
	bool    *d_finalizedVertices;           cudaMalloc(&d_finalizedVertices, sizeof(bool)   * graph->numVertices);
	unsigned int   *d_shortestDistances;           cudaMalloc(&d_shortestDistances, sizeof(unsigned int) * graph->numVertices);
	unsigned int   *d_updatingShortestDistances;   cudaMalloc(&d_updatingShortestDistances, sizeof(unsigned int) * graph->numVertices);
	bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * graph->numVertices);

	int neighbors = graph->numEdges / graph->numVertices;
	int iterator = NUM_ASYNCHRONOUS_ITERATIONS + (ASYNC_THRESHOLD / graph->numVertices) + (ASYNC_THRESHOLD_NEIGHBORS / neighbors);

	// --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
	initializeArrays << <ceil((float)(graph->numVertices)/BLOCK_SIZE), BLOCK_SIZE >> > (d_finalizedVertices, d_shortestDistances,
		d_updatingShortestDistances, sourceVertex, graph->numVertices);
	cudaDeviceSynchronize();

	// --- Read mask array from device -> host
	cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);

	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);

	cudaEventRecord(start0, 0);

	while (!allFinalizedVertices(h_finalizedVertices, graph->numVertices)) {

		// --- In order to improve performance, we run some number of iterations without reading the results.  This might result
		//     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
		//     stalling of the GPU waiting for results.
		for (int asyncIter = 0; asyncIter < iterator; asyncIter++) {

			Kernel1 << <(ceil((float)(graph->numVertices)/BLOCK_SIZE)), BLOCK_SIZE >> > (d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances,
				d_updatingShortestDistances, graph->numVertices, graph->numEdges);

			cudaDeviceSynchronize();
			Kernel2 << <(ceil((float)(graph->numVertices) / BLOCK_SIZE)), BLOCK_SIZE >> > (d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances,
				graph->numVertices);
			cudaDeviceSynchronize();
		}
		cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);
	}

	// --- Copy the result to host
	cudaMemcpy(h_shortestDistances, d_shortestDistances, sizeof(int) * graph->numVertices, cudaMemcpyDeviceToHost);	


	// Results timing section
	cudaEventRecord(stop1, 0);
	cudaEventRecord(stop0, 0);
	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop0);

	cudaEventElapsedTime(&elapsed, start1, stop1);
	cudaEventElapsedTime(&elapsedComp, start0, stop0);
	cudaEventDestroy(start1);
	cudaEventDestroy(start0);
	cudaEventDestroy(stop1);
	cudaEventDestroy(stop0);
	printf("The elapsed memory read time in gpu was %.2f ms\n", elapsed - elapsedComp);
	printf("The elapsed time in gpu computation only was %.2f ms\n", elapsedComp);
	// END of GPU timing

	free(h_finalizedVertices);

	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_weightArray);
	cudaFree(d_finalizedVertices);
	cudaFree(d_shortestDistances);
	cudaFree(d_updatingShortestDistances);
}

/****************/
/* MAIN PROGRAM */
/****************/
int main() {

	// --- Number of graph vertices
	int numVertices = 45000;
	// --- Number of edges per graph vertex
	int neighborsPerVertex = 50;

	// --- Source vertex
	int sourceVertex = 0;

	// --- Allocate memory for arrays
	GraphData graph;
	generateRandomGraph(&graph, numVertices, neighborsPerVertex);

	unsigned int *weightMatrix;
	unsigned int *h_shortestDistancesCPU = (unsigned int *)malloc(numVertices * sizeof(unsigned int));
	// --- From adjacency list to adjacency matrix.
	//     Initializing the adjacency matrix
	if (numVertices < 45001) // Prevent overflow for cpu graph
	{
		weightMatrix = (unsigned int *)malloc(numVertices * numVertices * sizeof(unsigned int));
		for (int k = 0; k < numVertices * numVertices; k++) weightMatrix[k] = INT_MAX;

		// --- Displaying the adjacency list and constructing the adjacency matrix
		printf("Adjacency list\n");
		for (int k = 0; k < numVertices; k++) weightMatrix[k * numVertices + k] = 0;
		for (int k = 0; k < numVertices; k++) {
			for (int l = 0; l < neighborsPerVertex; l++) {
				weightMatrix[k * numVertices + graph.edgeArray[graph.vertexArray[k] + l]] = graph.weightArray[graph.vertexArray[k] + l];
				if (numVertices < 100)
				{
					printf("Vertex nr. %i; Edge nr. %i; Weight = %d\n", k, graph.edgeArray[graph.vertexArray[k] + l],
						graph.weightArray[graph.vertexArray[k] + l]);
				}
			}
		}
		// --- Running Dijkstra on the CPU
		h_shortestDistancesCPU = (unsigned int *)malloc(numVertices * sizeof(unsigned int));

		// Timing CPU computation
		clock_t cpu_startTime, cpu_endTime;
		double cpu_ElapseTime = 0;

		cpu_startTime = clock();

		dijkstraCPU(weightMatrix, h_shortestDistancesCPU, sourceVertex, numVertices);

		cpu_endTime = clock();

		cpu_ElapseTime = ((cpu_endTime - cpu_startTime) / CLOCKS_PER_SEC);

		printf("CPU computation time: %.2f ms\n", cpu_ElapseTime);
		// END CPU Timing

		if (numVertices < 100) // too many results to be displayed
		{
			printf("\nCPU results\n");
			for (int k = 0; k < numVertices; k++)
			{
				if (h_shortestDistancesCPU[k] != INT_MAX)
				{
					printf("From vertex %i to vertex %i = %d\n", sourceVertex, k, h_shortestDistancesCPU[k]);
				}
				else
				{
					printf("From vertex %i to vertex %i = NO PATH\n", sourceVertex, k);
				}
			}
		}	
	}
	// --- Allocate space for the h_shortestDistancesGPU
	unsigned int *h_shortestDistancesGPU = (unsigned int*)malloc(sizeof(unsigned int) * graph.numVertices);
	dijkstraGPU(&graph, sourceVertex, h_shortestDistancesGPU); // contains timer
	if (numVertices < 100)
	{
		printf("\nGPU results\n");
		for (int k = 0; k < numVertices; k++)
		{
			if (h_shortestDistancesGPU[k] != INT_MAX)
			{
				printf("From vertex %i to vertex %i = %d\n", sourceVertex, k, h_shortestDistancesGPU[k]);
			}
			else
			{
				printf("From vertex %i to vertex %i = NO PATH\n", sourceVertex, k);
			}
		}
	}

	// Mismatch checking
	bool matching = true;
	unsigned int wrong = 0;
	for (int k = 0; k < numVertices; k++)
	{
		if (h_shortestDistancesGPU[k] != h_shortestDistancesCPU[k])
		{
			//printf("vertex mismatch = %d| CPU val = %d | GPU val = %d | Difference = %d\n", k, h_shortestDistancesCPU[k], h_shortestDistancesGPU[k], h_shortestDistancesCPU[k] - h_shortestDistancesGPU[k]);
			wrong++;
			matching = false;
		}
	}

	if (matching)
	{
		printf("CPU and GPU Cost arrays are matching!");
	}
	else if(!matching && numVertices < 45001)
	{
		printf("CPU and GPU Cost arrays DO NOT match\n");
		printf("%d mismatches." , wrong);
	}
	else
	{
		printf("CPU code could not be run.");
	}

	free(h_shortestDistancesCPU);
	free(h_shortestDistancesGPU);

	return 0;
}
