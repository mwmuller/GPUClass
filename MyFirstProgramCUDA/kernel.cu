
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;
#include <limits.h>

// Number of vertices in the graph
#define V 9

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
__global__ void minDistance(int dist[], bool sptSet[], int* &min_index)
{

    // Initialize min value
    int min = INT_MAX;

    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = new int(v);
}


__global__ void print2dArr(int* arr[V])
{
    int i = threadIdx.x;


}
__global__ void calcShortest(int graph[V][V], int* dist, bool* sptSet, int src)
{
    // Initialize all distances as INFINITE and stpSet[] as false

    int thread = threadIdx.x; // for each thread, init each node to infinte
        dist[thread] = INT_MAX, sptSet[thread] = false;

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        int u = 0;

        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in the first iteration.
        //minDistance(dist, sptSet, u);

        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist[v] only if is not in sptSet, there is an edge from
        // u to v, and total weight of path from src to  v through u is
        // smaller than current value of dist[v]
        if (!sptSet[thread] && graph[u][thread] && dist[u] != INT_MAX
            && dist[u] + graph[u][thread] < dist[thread])
            dist[thread] = dist[u] + graph[u][thread];
    }
}

// A utility function to print the constructed distance array
__global__ void printSolution(int* dist)
{
    printf("Vertex \t Distance from Source\n");
    int i = threadIdx.x;
        printf(" \t\t%d\n", dist[i]);
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
int* dijkstra(int graph[V][V], int src)
{
    cudaError_t cudaStatus;
    // init an array of ptrs to more arrays
    int* dev_graph;
    int* test_gr;

    memcpy(dev_graph, graph, V * V * sizeof(int));
    int* dist = { 0 }; // The output array.  dist[i] will hold the shortest
    int* dev_Dist = { 0 };
    // distance from src to i

    bool* sptSet = new bool[V]; // sptSet[i] will be true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized

    // Alloc and then copy ptr to 
    cudaMalloc((void**)&dev_Dist, V * sizeof(int));
    cudaMemcpy(dev_Dist, dist, V * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_graph, V * V * sizeof(int));
    cudaMemcpy(dev_graph, graph[0], V * V * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&test_gr, V * V * sizeof(int));
    cudaMemcpy(test_gr, dev_graph[0], V * V * sizeof(int), cudaMemcpyDeviceToHost);

    calcShortest<<<1, V>>>(dev_graph[0], dev_Dist, sptSet, src);

    cudaMemcpy(dist, dev_Dist, V * V * sizeof(int), cudaMemcpyDeviceToHost);

    return dist;
}

// driver program to test above function
int main()
{
    int* hostDist = new int[V];

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

    hostDist = dijkstra(graph, 0);

    printSolution<<<1, V>>>(hostDist);

    return 0;
}

// This code is contributed by shivanisinghss2110