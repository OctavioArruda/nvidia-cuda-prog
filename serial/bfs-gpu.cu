#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <list>

using namespace std;

#define INF 99999
#define GOAL 5000

__global__ void kernel_cuda_simple(int *v_adj_list, int *v_adj_length, int *v_adj_begin, int *result)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	// For every vertex
	for (int v = 0; v < num_vertices; v += num_threads)
	{
		int vertex = v + tid;
		if (vertex < num_vertices)
		{	
			// For every neightboor
			for (int n = 0; n < v_adj_length[vertex]; n++)
			{
				// Pega o vizinho através da lista de lista de adj_begin, passando o vértice mais n
				int neighbor = v_adj_list[v_adj_begin[vertex] + n];
				
				// se o result vizinho for maior que o result vertex + 1, sweepa
				if (result[neighbor] > result[vertex] + 1)
				{
					result[neighbor] = result[vertex] + 1;
					*still_running = true;
				}
			}
		}
	}
}

void run(int* v_adj_list, int* v_adj_length,int *v_adj_begin, int* result)
{ /* Setup and data transfer code omitted */
	while (*still_running)
	{	
		/* 
		For each iteration, sendo a new adj nodes list, length of adj nodes list, 
		the begin of each os these lists and a result
		*/
		cudaMemcpy(k_still_running, &false_value, sizeof(bool) * 1, cudaMemcpyHostToDevice);
		kernel_cuda_simple<<<BLOCKS, THREADS>>>(v_adj_list, v_adj_length, v_adj_begin, result);
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
	
	// Adjacency list for kernel:
	list<int> v_adj_list[nNodes];
	list<int> v_adj_begin;
	list<int> result;
	int offsetcount = 0;
	 
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
		
		int length_adj = 0;

        while (inputfile >> a >> b >> w)
        {	
			// Problema: como ler essa entrada já como uma lista de adjacencia?
			// Old implementation as adjacency matrix
			// edge, vertice = weigth			
            //graph[a][b] = w;
			//graph[b][a] = w;

			/*
			adjacency list be like:
			where a and b are list of neighboors of themselves			
			*/

			// Armazena o nodo adjacente e o peso
			v_adj_list[a].append(b);	
			v_adj_list[b].append(a);

			if (a != temp)
			{
				v_adj_length.append(length_adj);
				v_adj_begin.append(offsetcount);
				length_adj = 0;
			}
			
			// Just to check if we changed the vertice
			int temp = a;
					
			// this marks the length
			++length;
			
			// this will mark the start and the end of and adjacency
			++offsetcount;
        }
	}
	
	/*
	To do: 
	Graph shoulde be represented as a list of adjacency stored as v_adj_list
	after that, 
	*/

	int blockSize = 32;
	int numBlocks = (nNodes + blockSize - 1) / blockSize;

	run<<<numBlocks, blockSize>>>(int *v_adj_list, int *v_adj_length, int *v_adj_begin, int *result);

	printf("%d blocks of %d threads\n", numBlocks, blockSize);

	bool *visited = new bool[nNodes];

}
