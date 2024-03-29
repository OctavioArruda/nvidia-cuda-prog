#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define INF 99999
#define GOAL 5000

#define DEBUG 0

double GetTime(void)
{
   struct  timeval time;
   double  Time;
   
   gettimeofday(&time, (struct timezone *) NULL);
   Time = ((double)time.tv_sec*1000000.0 + (double)time.tv_usec);
   return(Time);
}

__global__ void bfs_kernel(int nNodes, int *graph, bool *visited, bool *isRunning) {
    int threadIdentifier = blockIdx.x * blockDim.x + threadIdx.x;
    int nThread = blockDim.x * gridDim.x;

	for (int v = 0; v < nNodes; v += nThread) 
	{
		int vertex = v + threadIdentifier;
		
		if (vertex < nNodes) 
		{
			for (int i = 0; i < nNodes; i++) 
			{
				if (graph[vertex * nNodes + i] != INF && vertex != i) 
				{
                    // Neighbor
					if (!visited[i]) 
					{
                        visited[i] = true;
                        *isRunning = true;
                    }
                }
            }
        }
    }
}


int main(int argc, char **argv){

	double timeElapsed, clockBegin;
	int **graph;
    int a, b, w, nNodes;

	if (argc < 2) 
	{
        cout << "Usage: ./" << argv[0] << " <graph>" << endl;
        exit(-1);
	}
	    
    ifstream inputfile(argv[1]);
    inputfile >> nNodes;
	graph = new int*[nNodes];
	
    for (int i = 0; i < nNodes; ++i)
    {
		graph[i] = new int[nNodes]; 
		
        for (int j = 0; j < nNodes; ++j)
        {
			graph[i][j] = INF;
		}
    }
    while (inputfile >> a >> b >> w)
    {
        graph[a][b] = w;
        graph[b][a] = w;
    }

    /* BFS */
    int *graph_d;
    bool false_val = false;
    bool *running, *running_d;
    bool *visited, *visited_d;

    cudaMalloc(&graph_d, nNodes * nNodes * sizeof(int));
    for (int i = 0; i < nNodes; i++)
	{
		cudaMemcpy(&graph_d[i * nNodes], graph[i], nNodes * sizeof(int), cudaMemcpyHostToDevice);
	}

    running = new bool[1];
    *running = true;
    cudaMalloc(&running_d, 1 * sizeof(bool));

	visited = new bool[nNodes];
	
    for(int i = 0; i < nNodes; i++)
        {
			visited[i] = 0;
		}
    cudaMalloc(&visited_d, nNodes * sizeof(bool));
    cudaMemcpy(visited_d, visited, nNodes * sizeof(bool), cudaMemcpyHostToDevice);

    clockBegin = GetTime();
    
	while (*running) 
	{
        cudaMemcpy(running_d, &false_val, 1 * sizeof(bool), cudaMemcpyHostToDevice);
        int blockSize = 32;
        int numBlocks = (nNodes + blockSize - 1) / blockSize;
        bfs_kernel<<<numBlocks, blockSize>>>(nNodes, graph_d, visited_d, running_d);
        cudaMemcpy(running, running_d, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    timeElapsed = (GetTime() - clockBegin)/1000000;
    
	cudaMemcpy(visited, visited_d, nNodes * sizeof(bool), cudaMemcpyDeviceToHost);
	
    #if DEBUG == 1
		for (int i = 0; i < nNodes; i++)
		{
            cout << "node " << i << ": " << visited[i] << endl;
		}
    #endif
        
    printf("%5lf\n", timeElapsed);

	for(int i = 0; i < nNodes; i++)
	{
		free(graph[i]);
	}

	free(visited);	
	free(running);
	
    cudaFree(visited_d);
    cudaFree(running_d);
    cudaFree(graph_d);
}

