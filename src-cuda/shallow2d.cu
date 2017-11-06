#include <string.h>
#include <math.h>

#ifndef RESTRICT
#define restrict __restrict__
#endif /* RESTRICT */
//ldoc on
/**
 * ## Implementation
 *
 * The actually work of computing the fluxes and speeds is done
 * by local (`static`) helper functions that take as arguments
 * pointers to all the individual fields.  This is helpful to the
 * compilers, since by specifying the `restrict` keyword, we are
 * promising that we will not access the field data through the
 * wrong pointer.  This lets the compiler do a better job with
 * vectorization.
 */

__constant__ static const float g = 9.8;


// total number of cells (ncells) = nx_all * ny_all
__global__ static 
void shallow2dv_flux(float* restrict fh,
                     float* restrict fhu,
                     float* restrict fhv,
                     float* restrict gh,
                     float* restrict ghu,
                     float* restrict ghv,
                     const float* restrict h,
                     const float* restrict hu,
                     const float* restrict hv,
                     float g)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    // linearize to 1D
    const unsigned int tid = ((gridDim.x * blockDim.x) * idy) + idx;
    
    float hi = h[tid], hui = hu[tid], hvi = hv[tid];
    float inv_h = 1/hi;
    fhu[tid] = hui*hui*inv_h + (0.5f*g)*hi*hi;
    fhv[tid] = hui*hvi*inv_h;
    ghu[tid] = hui*hvi*inv_h;
    ghv[tid] = hvi*hvi*inv_h + (0.5f*g)*hi*hi;
}


__global__  static
void shallow2dv_speed(float* restrict cxy,
                      const float* restrict h,
                      const float* restrict hu,
                      const float* restrict hv,
                      float g)
{
    float cx = cxy[0];
    float cy = cxy[1];
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    // linearize to 1D
    const unsigned int tid = ((gridDim.x * blockDim.x) * idy) + idx;
    float hi = h[tid];
    float inv_hi = 1.0f/h[tid];
    float root_gh = sqrtf(g * hi);
    float cxi = fabsf(hu[tid] * inv_hi) + root_gh;
    float cyi = fabsf(hv[tid] * inv_hi) + root_gh;
    if (cx < cxi) cx = cxi;
    if (cy < cyi) cy = cyi;
    cxy[0] = cx;
    cxy[1] = cy;
}

extern "C"
void shallow2d_flux_cu(float* FU, float* GU, const float* U,
                    int nx, int ny, int field_stride)
{
    cudaMemcpy(FU, U+field_stride,   nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(GU, U+2*field_stride, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    shallow2dv_flux<<<(nx * ny)/256, 256>>>(FU, FU+field_stride, FU+2*field_stride,
                    GU, GU+field_stride, GU+2*field_stride,
                    U,  U +field_stride, U +2*field_stride,
                    g);
}

extern "C"
void shallow2d_speed_cu(float* cxy, const float* U,
                     int nx, int ny, int field_stride)
{
    shallow2dv_speed<<<(nx * ny)/256, 256>>>(cxy, U, U+field_stride, U+2*field_stride, g);
}
