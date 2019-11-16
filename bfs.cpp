#include <iostream>
#include <fstream>
#include <stack>
#include <queue>
#include <sys/time.h>

using namespace std;

#define INF 99999
#define GOAL 5000

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
                graph[i][j] = INF;
        }
        while (inputfile >> a >> b >> w)
        {
            graph[a][b] = w;
            graph[b][a] = w;
        }
    }
	
	/*
	Utilizando memória unificada(mais fácil):
	int *A;
	cudaMallocManaged(&A, 1024 * sizeof(int));
	Qual tamanho e tipo que precisamos alocar?

	Dados do enunciado do trabalho:
	São fornecidos grafos de diferentes topologias para realização dos testes. São eles:

	binomial: árvore binomial de grau 12
	complete: grafo completo de 5 mil vértices
	random250: grafo aleatório de 5 mil vértices e 250 mil arestas
	random1250: grafo aleatório de 5 mil vértices e 1250000 arestas

	Logo seria algo do tipo:
	int *A;
	cudaMallocManaged(&A, 5000 * sizeof(int));
	OBS: No BFS, procura-se sempre o vértice 5000.

	Ideia de paralelização do BFS: Iniciar a busca por vários nodos diferentes ao mesmo
	tempo?

	*/
	bool *visited = new bool[nNodes];
	/* BFS */

	// paralelizavel
	for(int i = 0; i < nNodes; i++)
		visited[i] = false;
	// end

	/* Which node to visit next in a queue */
   	queue<int> q;
	q.push(0);
	visited[0] = true;
	
	clockBegin = GetTime();

	/* BFS */
	while(!q.empty()){

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

		/* paralelizável: partir de vários nodos ao mesmo tempo */
		for(int i = 0; i < nNodes; i++)
		{
			if(graph[v][i] != INF && v != i)
			{
				if(visited[i] == false)
				{
					visited[i] = true;
					q.push(i);
				}
			}
		}	
	}

	/* Talvez seja válido paralelizar os free... */
	for(int i = 0; i < nNodes; i++)
		free(graph[i]);
	free(visited);
}

