#include <fstream>
#include <iostream>
// C++ Program for Floyd Warshall Algorithm  
//#include <bits/stdc++.h> 
#include <sys/time.h>
using namespace std; 
  
/* Define Infinite as a large enough 
value.This value will be used for  
vertices not connected to each other */
#define INF 99999  
  
// A function to print the solution matrix  
void printSolution(int ** dist, int nNodes);  
__global__ void floydWarshall(int *dist, int nNodes, int k);

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
    int** dist;
    
    int* dist_1d;
    int* d_dist_1d;

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
				graph[i][j] = INF;	
		}
		while (inputfile >> a >> b >> w)	
		{
			graph[a][b] = w;
			graph[b][a] = w;
		}
	}
  
	dist = new int*[nNodes];
	for (int i = 0; i < nNodes; ++i)
		dist[i] = new int[nNodes];
    /* dist[][] will be the output matrix  
    that will finally have the shortest  
    distances between every pair of vertices */
  
    /* Initialize the solution matrix same  
    as input graph matrix. Or we can say  
    the initial values of shortest distances 
    are based on shortest paths considering  
    no intermediate vertex. */
    for (int i = 0; i < nNodes; i++)  
        for (int j = 0; j < nNodes; j++)
            dist[i][j] = graph[i][j];

    





    clockBegin = GetTime();
    

    // Passa a matriz 2D pra dentro de uma 1D
    dist_1d = (int *)malloc(nNodes*nNodes*sizeof(int));  //Pass the values into the subtable dist_1d
    int index = 0;
    for (int j = 0; j<nNodes; j++){
        for (int i = 0; i<nNodes; i++){
            dist_1d[index++] = dist[i][j];
        }
    }

    int size = nNodes*nNodes*sizeof(int);

    // Copia para o device
    cudaMalloc((void**)&d_dist_1d, size);  //Allocation of device Table
    cudaMemcpy(d_dist_1d, dist_1d, size, cudaMemcpyHostToDevice); //Memory transfer from host to device

    int blockSize = 32;
    int numBlocks = (nNodes + blockSize - 1) / blockSize;

    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(blockSize, blockSize);

    // lansa a braba k vezes
    for (int k = 0; k < nNodes; k++){		
        floydWarshall<<<dimGrid, dimBlock>>>(d_dist_1d, nNodes, k);  //Run kernel for each k
    }
    
    cudaMemcpy(dist_1d, d_dist_1d, size, cudaMemcpyDeviceToHost);  //Pass values from device to host
    cudaFree(d_dist_1d);

    // Passa devolta pra 2D
    index = 0;
    for (int j = 0; j < nNodes; j++){
        for (int i = 0; i < nNodes; i++){
            dist[i][j] = dist_1d[index++];  //Pass the values to the 2-d Table of min distance D[i][j]
        }
    }

	timeElapsed = (GetTime() - clockBegin)/1000000;







    // Print the shortest distance matrix  
    printSolution(dist, nNodes);  

    printf("Computation time: %5lf\n", timeElapsed);


    return 0;  
}  
  
// Solves the all-pairs shortest path  
// problem using Floyd Warshall algorithm  
__global__ void floydWarshall(int *dist, int nNodes, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;   //We find i & j in the Grid of threads
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > nNodes)
        return;

    if(j > nNodes)
        return;
    
    if(dist[i + j*nNodes] > dist[i + k*nNodes] + dist[k + j*nNodes]){
        dist[i + j*nNodes] = dist[i + k*nNodes] + dist[k + j*nNodes];  //Every thread calculates its proper value
    }
}
  
/* A utility function to print solution */
void printSolution(int **dist, int nNodes)  
{  
    for (int i = 0; i < nNodes; i++)  
    {  
        for (int j = 0; j < nNodes; j++)  
        {  
            if (dist[i][j] == INF)  
                cout<<"INF"<<"     ";  
            else
                cout<<dist[i][j]<<"     ";  
        }  
        cout<<endl;  
    }  
}  
  
// This code is contributed by rathbhupendra 

