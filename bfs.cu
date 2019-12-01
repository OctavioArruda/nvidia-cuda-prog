#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define INF 99999
#define GOAL 5000

__global__ void kernel_cuda_simple(...)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	for (int v = 0; v < num_vertices; v += num_threads)
	{
		int vertex = v + tid;
		if (vertex < num_vertices)
		{
			for (int n = 0; n < v_adj_length[vertex]; n++)
			{
				int neighbor = v_adj_list[v_adj_begin[vertex] + n];

				if (result[neighbor] > result[vertex] + 1)
				{
					result[neighbor] = result[vertex] + 1;
					*still_running = true;
				}
			}
		}
	}
}
void run()
{ /* Setup and data transfer code omitted */
	while (*still_running)
	{
		cudaMemcpy(k_still_running, &false_value, sizeof(bool) * 1, cudaMemcpyHostToDevice);
		kernel_cuda_simple<<<BLOCKS, THREADS>>>(...);
		cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
	}
	
	cudaThreadSynchronize();
}

double GetTime(void)
{
   struct  timeval time;
   double  Time;

   gettimeofday(&time, (struct timezone *) NULL);
   Time = ((double)time.tv_sec*1000000.0 + (double)time.tv_usec);
   return(Time);
}


int main(int argc, char **argv){
	double timeElapsed, clockBegin;
	int** graph;
    int a, b, w, nNodes;

    if (argc > 1)
    {
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
    }

	int blockSize = 32;
	int numBlocks = (nNodes + blockSize - 1) / blockSize;
	printf("%d blocks of %d threads\n", numBlocks, blockSize);

	bool *visited = new bool[nNodes];

	initVisitedVec<<<numBlocks, blockSize>>>(visited);
	cudaDeviceSynchronize();

	searchParallel<<<numBlocks, blockSize>>>(visited, graph);
	cudaDeviceSynchronize();

	freeNodes<<<numBlocks, blockSize>>>(visited, graph);
	cudaDeviceSynchronize();
}

