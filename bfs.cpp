#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define INF 99999
#define GOAL 5000

/*	Init visited vec as false parallel */
__global__ void initVisitedVec(bool *Vec)
{
	int idx;
	int sizeVec = Vec.length();

	for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeVec; idx += blockDim.x * gridDim.x)
	{
		Vec[idx] = false;
	}

}

__global__ void freeNodes(int *visited, int** *graph)
{
	int idx;
	int sizeVisited = visited.length();

	for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeVisited; idx += blockDim.x * gridDim.x)
	{
		cudaFree(visited);
		cudaFree(graph[idx]);
	}
}

/* Paralellel search*/
__global__ void searchParallel(int *visited, int** *graph)
{
	int idx;
	int sizeVisited = visited.length();
	queue<int> q;
	q.push(0);
	visited[0] = true;
	
	clockBegin = GetTime();

	while(!q.empty())
	{

		/* first node */
		int v = q.front();
		q.pop();
		cout << "visited " << v << endl;
		if(v == GOAL)
		{
			cout << "Found " << GOAL << endl;
			timeElapsed = (GetTime() - clockBegin)/1000000;
			printf("Computation time: %5lf\n", timeElapsed);
			return 0;
		}

		for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeVisited; idx += blockDim.x * gridDim.x)
		{
			if(graph[v][idx] != INF && v != idx)
			{
				if(visited[idx] == false)
				{
					visited[idx] = true;
					q.push(idx);
				}
			}
		}
	}
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

