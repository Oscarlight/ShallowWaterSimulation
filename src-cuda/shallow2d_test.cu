#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
extern "C" {
#include "shallow2d.cuh"
#include "shallow2d_base.h"
}

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int nx, int ny, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int nx, int ny, int field_stride);

void testShallow2d_by_pointer(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride,
                flux_t flux, speed_t speed){
	flux(FU, GU, U, nx, ny, field_stride);
	speed(cxy, U, nx, ny, field_stride);
}

void testShallow2d_baseline(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride){
	shallow2d_flux(FU, GU, U, nx*ny, field_stride);
	shallow2d_speed(cxy, U, nx*ny, field_stride);
}

void print_array(float* array, int len) {
	for(int i = 0; i < len; i++) {
	    printf("%.2f ", array[i]);    
	}
	printf("\n");
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv){
	if(argc<=2) {
        printf("You did not feed me arguments, I will die now :( ...");
        exit(1);
    }
	cudaEvent_t start,stop;
	float ms;
	const int nx = atoi(argv[1]), ny = atoi(argv[2]);
	const int ncell = nx * ny;
	const int field_stride = nx * ny;
	float cxy[2] = {1.0, 2.0};
	float FU[ncell * 3], GU[ncell * 3], U[ncell * 3];
	int i;
	for (i = 0; i < ncell * 3; i++) {
    	FU[i] = 1; GU[i] = 1; U[i] = 1;
	}
	// print_array(FU, ncell*3);

	// Execute baseline code
	struct timeval startc, end;
 	long seconds, useconds;
 	double mtime;
    gettimeofday(&startc, NULL);
	testShallow2d_baseline(cxy, FU, GU, U, nx, ny, field_stride);
	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - startc.tv_sec;
	useconds = end.tv_usec - startc.tv_usec;
	mtime = useconds;
	mtime/=1000;
	mtime+=seconds*1000;
    printf("CPU: %g ms. \n",mtime);

    // save true values
	float tFU[ncell * 3], tGU[ncell * 3], tU[ncell * 3];
	for (i = 0; i < ncell * 3; i++) {
    	tFU[i] = FU[i]; tGU[i] = GU[i]; tU[i] = U[i];
	}

	// Reset
	for (i = 0; i < ncell * 3; i++) {
    	FU[i] = 1; GU[i] = 1; U[i] = 1;
	}

	// Execute on GPU: using function pointer
	bool UNIFIED_MEM = true;
	// device copies of FU, GU, U
    float *dev_FU, *dev_GU, *dev_U, *dev_cxy;
    int size = ncell*3*sizeof(float);
    if (!UNIFIED_MEM) {
	    gpuErrchk(cudaMalloc( (void**)&dev_FU, size ));
	    gpuErrchk(cudaMalloc( (void**)&dev_GU, size ));
	    gpuErrchk(cudaMalloc( (void**)&dev_U,  size ));
	    gpuErrchk(cudaMalloc( (void**)&dev_cxy, 2*sizeof(float) ));
		copy the reseted data to GPU
		gpuErrchk(cudaMemcpy( dev_FU, FU, size, cudaMemcpyHostToDevice ));
	    gpuErrchk(cudaMemcpy( dev_GU, GU, size, cudaMemcpyHostToDevice ));
	    gpuErrchk(cudaMemcpy( dev_U,  U,  size, cudaMemcpyHostToDevice ));
	    gpuErrchk(cudaMemcpy( dev_cxy, cxy, 2*sizeof(float), cudaMemcpyHostToDevice ));
	} else {
	    // CUDA Unified memory
	    cudaMallocManaged(&dev_FU, size);
	    cudaMallocManaged(&dev_GU, size);
	    cudaMallocManaged(&dev_U, size);
	    cudaMallocManaged(&dev_cxy, 2*sizeof(float));
	   	// Reset
		for (i = 0; i < ncell * 3; i++) {
	    	dev_FU[i] = 1; dev_GU[i] = 1; dev_U[i] = 1;
		}
	} 

    // Time the GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	testShallow2d_by_pointer(
		dev_cxy, dev_FU, dev_GU, dev_U, nx, ny, field_stride,
		shallow2d_flux_cu,
		shallow2d_speed_cu
	);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU: %f ms. \n",ms);

	if (!UNIFIED_MEM) {
		cudaMemcpy( FU, dev_FU, size, cudaMemcpyDeviceToHost );
		cudaMemcpy( GU, dev_GU, size, cudaMemcpyDeviceToHost );
		cudaMemcpy( U,  dev_U,  size, cudaMemcpyDeviceToHost );	
	}

	printf("GPUassert: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("Check correctness ");
	for (i = 0; i < ncell * 3; i++)
		if (!UNIFIED_MEM) {
	    	if (FU[i] != tFU[i] or GU[i] != tGU[i] or U[i] != tU[i])
	    		printf("Wrong! \n");
		} else {
	    	if (dev_FU[i] != tFU[i] or dev_GU[i] != tGU[i] or dev_U[i] != tU[i])
	    		printf("Wrong! \n");
	    }
   
	printf("\n");

	cudaFree( dev_FU );
	cudaFree( dev_GU );
	cudaFree( dev_U );
}