#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
extern "C" {
#include "stepper.cuh"
#include "stepper_base.h"
#include "shallow2d.cuh"
}

void print_array(float* array, int len) {
	for(int i = 0; i < len; i++) {
	    printf("%.2f ", array[i]);    
	}
	printf("\n");
}

int main(int argc, char** argv){
	int nx = 3, ny = 3, ng = 1, nfield = 2;
	int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
	int nc = nx_all * ny_all;
    int N  = nfield * nc;
  	float* u  = (float*) malloc((4*N + 6*nx_all)* sizeof(float));
  	float* u_ture  = (float*) malloc((4*N + 6*nx_all)* sizeof(float));
    float* v  = u + N;
    float* f  = u + 2*N;
    float* g  = u + 3*N;
    float* scratch = u + 4*N;
    srand(time(NULL));
    int i;
    for (i = 0; i < 4*N + 6*nx_all; i++) {
    	u[i] = cos((float)i/float(4*N + 6*nx_all))
    	u_ture[i] = u[i];
    }
    float dtcdx2 = 0.3, dtcdy2 = 0.3;
    print_array(u_ture + N + 4, 1);
	central2d_predict_base(v, scratch, u, f, g, dtcdx2, dtcdy2,
                  nx, ny, nfield);
	print_array(v + 4, 1);
	central2d_predict_base_linear(v, scratch, u, f, g, dtcdx2, dtcdy2,
              nx, ny, nfield);


}