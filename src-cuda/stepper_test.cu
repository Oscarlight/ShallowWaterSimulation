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
	int nx = 10, ny = 10, ng = 4, nfield = 4;
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
    // set
    int i;
    for (i = 0; i < 4*N + 6*nx_all; i++) {
    	u[i] = cos((float)i/float(4*N + 6*nx_all));
    }
    float dtcdx2 = 0.3, dtcdy2 = 0.3;

	central2d_predict_base(v, scratch, u, f, g, dtcdx2, dtcdy2,
                  nx, ny, nfield);

	// baseline result
	for (i = 0; i < 4*N + 6*nx_all; i++) {
    	u_ture[i] = u[i];
    }

	// reset
	printf("Test linearized series code. \n");
    for (i = 0; i < 4*N + 6*nx_all; i++) {
    	u[i] = cos((float)i/float(4*N + 6*nx_all));
    }

	central2d_predict_base_linear(v, scratch, u, f, g, dtcdx2, dtcdy2,
              nx, ny, nfield);

	printf("Check correctness\n");
	for (i = 0; i < 4*N + 6*nx_all; i++)
    	if (u[i] != u_ture[i])
    		printf("Wrong! \n");

    // reset
  	printf("Test GPU code. \n");
    for (i = 0; i < 4*N + 6*nx_all; i++) {
    	u[i] = cos((float)i/float(4*N + 6*nx_all));
    }  

    // 
    float *dev_u, *dev_v, *dev_f, *dev_g, *dev_scratch;
    cudaMalloc( (void**)&dev_u, N );
    cudaMalloc( (void**)&dev_v, N );
    cudaMalloc( (void**)&dev_f, N );
    cudaMalloc( (void**)&dev_g, N );
    cudaMalloc( (void**)&dev_scatch, 6*nx_all*sizeof(float) );

    cudaMemcpy( dev_v, v, N, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_f, f, N, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_g, g, N, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_scratch, scratch, 
      6*nx_all*sizeof(float), 
      cudaMemcpyHostToDevice
    );

    float* dev_dtcdx2, dev_dtcdy2;
    int* dev_nx_all, int* dev_ny_all, int* dev_nfield;
    cudaMalloc( (void**)&dev_dtcdx2, sizeof(float) );
    cudaMalloc( (void**)&dev_dtcdy2, sizeof(float) ); 
    cudaMalloc( (void**)&dev_nx_all, sizeof(int) );
    cudaMalloc( (void**)&dev_ny_all, sizeof(int) );
    cudaMalloc( (void**)&dev_nfield, sizeof(int) );

    cudaMemcpy(dev_dtcdx2, dtcdx2, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dtcdy2, dtcdy2, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nx_all, nx_all, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ny_all, ny_all, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nfield, nfield, sizeof(int), cudaMemcpyHostToDevice);

    central2d_predict(
    		dev_v,dev_scratch,dev_u,dev_f,dev_g,dev_dtcdx2,dev_dtcdy2,
            dev_nx_all,dev_ny_all,dev_nfield
    );

    cudaMemcpy( u, dev_u, N, cudaMemcpyDeviceToHost);
    cudaMemcpy( v, dev_v, N, cudaMemcpyDeviceToHost);
    cudaMemcpy( scatch, dev_scratch, 6*nx_all*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( f, dev_f, N, cudaMemcpyDeviceToHost);
    cudaMemcpy( g, dev_g, N, cudaMemcpyDeviceToHost);

}