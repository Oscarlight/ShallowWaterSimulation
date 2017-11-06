#ifndef STEPPER_H
#define STEPPER_H
#ifndef RESTRICT
#define restrict __restrict__
#endif /* RESTRICT */
#include <math.h>

typedef void (*flux_t_base)(float* FU, float* GU, const float* U,
                       int ncell, int field_stride);
typedef void (*speed_t_base)(float* cxy, const float* U,
                        int ncell, int field_stride);

typedef struct central2d_t_base {

    int nfield;   // Number of components in system
    int nx, ny;   // Grid resolution in x/y (without ghost cells)
    int ng;       // Number of ghost cells
    float dx, dy; // Cell width in x/y
    float cfl;    // Max allowed CFL number

    // Flux and speed functions
    flux_t_base flux;
    speed_t_base speed;

    // Storage
    float* u;
    float* v;
    float* f;
    float* g;
    float* scratch;

} central2d_t_base;

int central2d_run_base(central2d_t_base* sim, float tfinal);

void central2d_predict_base(
    float* restrict v, 
    float* restrict scratch,
    const float* restrict u,
    const float* restrict f,
    const float* restrict g,
    float dtcdx2, float dtcdy2,
    int nx, int ny, int nfield
);

void central2d_predict_base_linear(
    float* restrict v, 
    float* restrict scratch,
    const float* restrict u,
    const float* restrict f,
    const float* restrict g,
    float dtcdx2, float dtcdy2,
    int nx, int ny, int nfield
);

void central2d_correct_base(
    float* restrict v,
    float* restrict scratch,
    const float* restrict u,
    const float* restrict f,
    const float* restrict g,
    float dtcdx2, float dtcdy2,
    int xlo, int xhi, int ylo, int yhi,
    int nx, int ny, int nfield
);
//ldoc off
#endif /* STEPPER_H */
