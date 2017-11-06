#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
extern "C" {
#include "stepper.cuh"
#include "stepper.h"
}


int main(int argc, char** argv){
	int nx = 1, ny = 2, ng = 4, nfield = 3;
	int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
	int nc = nx_all * ny_all;
    int N  = nfield * nc;
  	float* u  = (float*) malloc((4*N + 6*nx_all)* sizeof(float));
    float* v  = u + N;
    float* f  = u + 2*N;
    float* g  = u + 3*N;
    float* scratch = u + 4*N
    int i;
    for (i = 0; i < 4*N + 6*nx_all; i++)
    	u[i] = 0.5;

    float dtcdx2 = 0.3, dtcdy2 = 0.3;
	central2d_predict(v, scratch, u, f, g, dtcdx2, dtcdy2,
                  nx, ny, nfield)
}