#include <stdio.h>
#include <stdlib.h>
#include "shallow2d.cuh"
#include "shallow2d.h"

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int nx, int ny, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int nx, int ny, int field_stride);

__global__ 
void testShallow2d_by_reference(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride){
	shallow2d_flux_cu(FU, GU, U, nx, ny, field_stride);
	shallow2d_speed_cu(cxy, U, nx, ny, field_stride);
}

void testShallow2d_baseline(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride){
	shallow2d_flux(FU, GU, U, nx*ny, field_stride);
	shallow2d_speed(cxy, U, nx*ny, field_stride);
}

int main(int argc, char** argv){
	const int nx = 1, ny = 2;
	const int ncell = nx * ny;
	const int field_stride = nx * ny;
	float cxy[2] = {1.0, 2.0};
	float FU[ncell * 3], GU[ncell * 3], U[ncell * 3];
	int i;
	for (i = 0; i < ncell * 3; i++) {
    	FU[i] = rand();
    	GU[i] = rand();
    	U[i] = rand();
	}

	// Execute baseline code
	testShallow2d_baseline(cxy, FU, GU, U, nx, ny, field_stride);
}