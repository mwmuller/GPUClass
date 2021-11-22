#include <sstream>
#include <vector>
#include <iostream>
#include <float.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

#define BLOCK_SIZE 16

#define floatScalar 10000


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
	float *weightArray;

} GraphData;

/**********************************/
/* GENERATE RANDOM GRAPH FUNCTION */
/**********************************/
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex) {

	graph->numVertices = numVertices;
	graph->vertexArray = (int *)malloc(graph->numVertices * sizeof(int));
	graph->numEdges = numVertices * neighborsPerVertex;
	graph->edgeArray = (int *)malloc(graph->numEdges * sizeof(int));
	graph->weightArray = (float *)malloc(graph->numEdges * sizeof(float));

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
			graph->weightArray[k * neighborsPerVertex + l] = (float)(rand() % 1000) / 1000.0f;
		}
	}
}

/************************/
/* minDistance FUNCTION */
/************************/
// --- Finds the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree
int minDistance(float *shortestDistances, bool *finalizedVertices, const int sourceVertex, const int N) {

	// --- Initialize minimum value
	int minIndex = sourceVertex;
	float min = FLT_MAX;

	for (int v = 0; v < N; v++)
		if (finalizedVertices[v] == false && shortestDistances[v] <= min) min = shortestDistances[v], minIndex = v;

	return minIndex;
}

/************************/
/* dijkstraCPU FUNCTION */ // This will remain unchanged
/************************/
void dijkstraCPU(float *graph, float *h_shortestDistances, int sourceVertex, const int N) {

	// --- h_finalizedVertices[i] is true if vertex i is included in the shortest path tree
	//     or the shortest distance from the source node to i is finalized
	bool *h_finalizedVertices = (bool *)malloc(N * sizeof(bool));

	// --- Initialize h_shortestDistancesances as infinite and h_shortestDistances as false
	for (int i = 0; i < N; i++) h_shortestDistances[i] = FLT_MAX, h_finalizedVertices[i] = false;

	// --- h_shortestDistancesance of the source vertex from itself is always 0
	h_shortestDistances[sourceVertex] = 0.f;

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
				h_shortestDistances[currentVertex] != FLT_MAX &&
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

	for (int i = 1; i < numVertices; i++)
	{
		if (finalizedVertices[i] == false)
		{
			printf("Index that is true: %d", i);
			return false;
		}
	}


	return true;
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, float* __restrict__ d_shortestDistances, float* __restrict__ d_updatingShortestDistances,
	const int sourceVertex, const int numVertices) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numVertices) {

		d_finalizedVertices[tid] = false;
		d_shortestDistances[tid] = FLT_MAX;
		d_updatingShortestDistances[tid] = FLT_MAX;

		/*
			if (sourceVertex == tid) {

				d_finalizedVertices[tid] = true;
				d_shortestDistances[tid] = 0.f;
				d_updatingShortestDistances[tid] = 0.f;
			}

			else {

				d_finalizedVertices[tid] = false;
				d_shortestDistances[tid] = FLT_MAX;
				d_updatingShortestDistances[tid] = FLT_MAX;
			}
			*/
	}
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel1(const int * __restrict__ vertexArray, const int* __restrict__ edgeArray,
	const float * __restrict__ weightArray, bool * __restrict__ finalizedVertices, float * __restrict__ shortestDistances,
	float * __restrict__ updatingShortestDistances, const int numVertices, const int numEdges, int * tempIntMinArray) {

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < numVertices) {

		if (finalizedVertices[tid] != true) {

			int edgeStart = tid * (numEdges / numVertices), edgeEnd; // get the edge index that we start at

			// Check if we are beyond the number of verticies that we can check
			if (tid + 1 < (numVertices)) edgeEnd = (tid + 1) * (numEdges / numVertices); // Check if we are in bounds. 
			else                         edgeEnd = numEdges; // We are at the max.

			for (int edge = edgeStart; edge < edgeEnd; edge++) {
				int nid = edgeArray[edge]; // get the ID which will be associated with a vertex
				tempIntMinArray[nid] = updatingShortestDistances[nid] * floatScalar; // Assign the current value into the temp array
				int shortWeightMin;
				if (shortestDistances[nid] == FLT_MAX) // We need to check if the edge has been processed
				{
					shortestDistances[nid] = 0;
				}
				
				shortWeightMin = ((shortestDistances[nid] * floatScalar) + (weightArray[edge] * floatScalar));

				atomicMin(&tempIntMinArray[nid], shortWeightMin); // assigns minimum value to uSD pointer

				__syncthreads(); // Sync before assigning updatingShortestInt version to float array.
				if (tempIntMinArray[nid] != 0 && tid < nid)
				{
					shortestDistances[nid] = shortWeightMin;
					updatingShortestDistances[nid] = (float)(tempIntMinArray[nid]) / floatScalar;
				}
			}
		}
	}
}

/**************************/
/* DIJKSTRA GPU KERNEL #2 */
/**************************/
__global__  void Kernel2(const int * __restrict__ vertexArray, const int * __restrict__ edgeArray, const float* __restrict__ weightArray,
	bool * __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
	const int numVertices) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numVertices) {

		if (shortestDistances[tid] > updatingShortestDistances[tid]) {
			shortestDistances[tid] = updatingShortestDistances[tid];
			finalizedVertices[tid] = true;
		}

		updatingShortestDistances[tid] = shortestDistances[tid];
	}
}

/************************/
/* dijkstraGPU FUNCTION */
/************************/
void dijkstraGPU(GraphData *graph, const int sourceVertex, float * __restrict__ h_shortestDistances) {

	// --- Create device-side adjacency-list, namely, vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
	int     *d_vertexArray;         cudaMalloc(&d_vertexArray, sizeof(int)   * graph->numVertices);
	int     *d_edgeArray;           cudaMalloc(&d_edgeArray, sizeof(int)   * graph->numEdges);
	float   *d_weightArray;         cudaMalloc(&d_weightArray, sizeof(float) * graph->numEdges);

	// --- Copy adjacency-list to the device
	cudaMemcpy(d_vertexArray, graph->vertexArray, sizeof(int)   * graph->numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgeArray, graph->edgeArray, sizeof(int)   * graph->numEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weightArray, graph->weightArray, sizeof(float) * graph->numEdges, cudaMemcpyHostToDevice);

	// --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
	bool    *d_finalizedVertices;           cudaMalloc(&d_finalizedVertices, sizeof(bool)   * graph->numVertices);
	float   *d_shortestDistances;           cudaMalloc(&d_shortestDistances, sizeof(float) * graph->numVertices);
	float   *d_updatingShortestDistances;   cudaMalloc(&d_updatingShortestDistances, sizeof(float) * graph->numVertices);
	int		*d_tempIntMinArray;				cudaMalloc(&d_tempIntMinArray, sizeof(int) * graph->numVertices);
	bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * graph->numVertices);

	// --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
	initializeArrays << <ceil((float)(graph->numVertices)/BLOCK_SIZE), BLOCK_SIZE >> > (d_finalizedVertices, d_shortestDistances,
		d_updatingShortestDistances, sourceVertex, graph->numVertices);
	//cudaPeekAtLastError());
	cudaDeviceSynchronize();

	// --- Read mask array from device -> host
	cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);

	//while (!allFinalizedVertices(h_finalizedVertices, graph->numVertices)) {

		// --- In order to improve performance, we run some number of iterations without reading the results.  This might result
		//     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
		//     stalling of the GPU waiting for results.
		for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++) {

			Kernel1 << <(ceil((float)(graph->numVertices)/BLOCK_SIZE)), BLOCK_SIZE >> > (d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances,
				d_updatingShortestDistances, graph->numVertices, graph->numEdges, d_tempIntMinArray);

			cudaDeviceSynchronize();
			Kernel2 << <(ceil((float)(graph->numVertices) / BLOCK_SIZE)), BLOCK_SIZE >> > (d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances,
				graph->numVertices);
			cudaDeviceSynchronize();
		}

		cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);
	//}

	// --- Copy the result to host
	cudaMemcpy(h_shortestDistances, d_shortestDistances, sizeof(float) * graph->numVertices, cudaMemcpyDeviceToHost);

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
	int numVertices = 8;

	// --- Number of edges per graph vertex
	int neighborsPerVertex = 6;

	// --- Source vertex
	int sourceVertex = 0;

	// --- Allocate memory for arrays
	GraphData graph;
	generateRandomGraph(&graph, numVertices, neighborsPerVertex);

	// --- From adjacency list to adjacency matrix.
	//     Initializing the adjacency matrix
	float *weightMatrix = (float *)malloc(numVertices * numVertices * sizeof(float));
	for (int k = 0; k < numVertices * numVertices; k++) weightMatrix[k] = FLT_MAX;

	// --- Displaying the adjacency list and constructing the adjacency matrix
	printf("Adjacency list\n");
	for (int k = 0; k < numVertices; k++) weightMatrix[k * numVertices + k] = 0.f;
	for (int k = 0; k < numVertices; k++)
		for (int l = 0; l < neighborsPerVertex; l++) {
			weightMatrix[k * numVertices + graph.edgeArray[graph.vertexArray[k] + l]] = graph.weightArray[graph.vertexArray[k] + l];
			printf("Vertex nr. %i; Edge nr. %i; Weight = %f\n", k, graph.edgeArray[graph.vertexArray[k] + l],
				graph.weightArray[graph.vertexArray[k] + l]);
		}

	for (int k = 0; k < numVertices * neighborsPerVertex; k++)
		printf("%i %i %f\n", k, graph.edgeArray[k], graph.weightArray[k]);
	
	// --- Displaying the adjacency matrix
	if (numVertices < 100)
	{
		printf("\nAdjacency matrix\n");
		for (int k = 0; k < numVertices; k++) {
			for (int l = 0; l < numVertices; l++)
				if (weightMatrix[k * numVertices + l] < FLT_MAX)
					printf("%1.3f\t", weightMatrix[k * numVertices + l]);
				else
					printf("--\t");
			printf("\n");
		}
	}
	else
	{
		// do nothing because we don't have that kind of time
	}
	// --- Running Dijkstra on the CPU
	float *h_shortestDistancesCPU = (float *)malloc(numVertices * sizeof(float));
	dijkstraCPU(weightMatrix, h_shortestDistancesCPU, sourceVertex, numVertices);

	printf("\nCPU results\n");
	for (int k = 0; k < numVertices; k++) printf("From vertex %i to vertex %i = %f\n", sourceVertex, k, h_shortestDistancesCPU[k]);

	// --- Allocate space for the h_shortestDistancesGPU
	float *h_shortestDistancesGPU = (float*)malloc(sizeof(float) * graph.numVertices);
	dijkstraGPU(&graph, sourceVertex, h_shortestDistancesGPU);

	printf("\nGPU results\n");
	for (int k = 0; k < numVertices; k++) printf("From vertex %i to vertex %i = %f\n", sourceVertex, k, h_shortestDistancesGPU[k]);

	free(h_shortestDistancesCPU);
	free(h_shortestDistancesGPU);

	return 0;
}