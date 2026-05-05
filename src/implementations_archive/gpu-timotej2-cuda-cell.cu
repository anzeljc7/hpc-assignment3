#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gifenc.h"
#include "lennard-jones.h"

/*
 * CUDA implementacija Lennard-Jones simulacije s cell-list optimizacijo.
 *
 * Cell-list oziroma linked-cell optimizacija:
 *    Simulacijski prostor razdelimo na kvadratne celice s stranico >= R_CUT.
 *    Vsaka CUDA nit obdela en delec i in pregleda samo 3x3 = 9 sosednjih
 *    celic namesto vseh N delcev. Tako se izognemo preverjanju vseh N^2 parov.
 *
 * Vsaka nit akumulira silo delca i v lokalnih spremenljivkah in jo ob koncu
 * zapiše samo v fx[i]/fy[i]. Ker nobena nit ne piše v tuje delce, ne
 * potrebujemo atomicAdd. Ker vsak par (i,j) obiščemo dvakrat (enkrat za i,
 * enkrat za j), upoštevamo faktor 0.5 pri potencialni energiji.
 *
 * V primerjavi z gpu-timotej2-cuda-newton-cell.cu:
 *    - Ni atomicAdd pri silah -> enostavnejši dostopi do pomnilnika
 *    - Vsak par obravnavan dvakrat (Newton bi prepolovil delo, a zahteva atomics)
 *    - Enostavnejša struktura: en blok delcev, ni para celic
 */

#ifndef VECTOR_THREADS
#define VECTOR_THREADS 32
#endif

#ifndef REDUCE_THREADS
#define REDUCE_THREADS 32
#endif

#ifndef MAX_PARTICLES_PER_CELL
#define MAX_PARTICLES_PER_CELL 64
#endif

#ifndef CHECK_CELL_OVERFLOW
#define CHECK_CELL_OVERFLOW 0
#endif

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess)                                                 \
        {                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_err));                                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#if GENERATE_GIF
uint8_t palette[] = {0, 0, 0, 255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index)
{
    if (x < 0 || y < 0 || x >= w || y >= h)
        return;
    img[(size_t)y * (size_t)w + (size_t)x] = index;
}

void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size)
{
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);
    for (unsigned int i = 0; i < n; ++i)
    {
        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;
        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy)
        {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx)
            {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS)
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
            }
        }
    }
}
#endif

// -----------------------------------------------------------------------------
// Pomožne funkcije za CPE in združljivost z osnovno kodo
// -----------------------------------------------------------------------------

double random_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

double compute_ke(const Particle *particles, unsigned int n)
{
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i)
    {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature)
{
    srand(seed);

    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;

    for (unsigned int k = 0; k < n; ++k)
    {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        particles[k].fx = 0.0;
        particles[k].fy = 0.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;

    double ke = 0.0;
    for (unsigned int k = 0; k < n; ++k)
    {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0)
        return 0;

    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; ++k)
    {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

void wrap_positions(Particle *particles, unsigned int n, double box_size)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);
        if (wx < 0.0) wx += box_size;
        if (wy < 0.0) wy += box_size;
        p->x = wx;
        p->y = wy;
    }
}

static __host__ __device__ inline double lj_v_shift(void)
{
    const double sr = SIGMA / R_CUT;
    const double sr2 = sr * sr;
    const double sr6 = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;
    return 4.0 * EPSILON * (sr12 - sr6);
}

double compute_v_shift(void)
{
    return lj_v_shift();
}

double compute_forces(Particle *particles, unsigned int n, double box_size)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = compute_v_shift();
    double pe = 0.0;

    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = i + 1; j < n; ++j)
        {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;

            if (dx > half_box) dx -= box_size;
            else if (dx < -half_box) dx += box_size;
            if (dy > half_box) dy -= box_size;
            else if (dy < -half_box) dy += box_size;

            double r2 = dx * dx + dy * dy;
            if (r2 < rcut2 && r2 > 0.0)
            {
                double inv_r2 = 1.0 / r2;
                double sr2 = (SIGMA * SIGMA) * inv_r2;
                double sr6 = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;
                double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                double fij_x = f_over_r * dx;
                double fij_y = f_over_r * dy;

                particles[i].fx += fij_x;
                particles[i].fy += fij_y;
                particles[j].fx -= fij_x;
                particles[j].fy -= fij_y;

                pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
            }
        }
    }

    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

// -----------------------------------------------------------------------------
// CUDA pomožne funkcije in kerneli
// -----------------------------------------------------------------------------

static __device__ __forceinline__ int wrap_cell(int c, int nc)
{
    if (c < 0) return c + nc;
    if (c >= nc) return c - nc;
    return c;
}

__global__ void pack_particles_kernel(const Particle *__restrict__ p,
                                      double *__restrict__ x,
                                      double *__restrict__ y,
                                      double *__restrict__ vx,
                                      double *__restrict__ vy,
                                      double *__restrict__ fx,
                                      double *__restrict__ fy,
                                      unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = p[i].x;
    y[i] = p[i].y;
    vx[i] = p[i].vx;
    vy[i] = p[i].vy;
    fx[i] = p[i].fx;
    fy[i] = p[i].fy;
}

__global__ void unpack_particles_kernel(Particle *__restrict__ p,
                                        const double *__restrict__ x,
                                        const double *__restrict__ y,
                                        const double *__restrict__ vx,
                                        const double *__restrict__ vy,
                                        const double *__restrict__ fx,
                                        const double *__restrict__ fy,
                                        unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].x = x[i];
    p[i].y = y[i];
    p[i].vx = vx[i];
    p[i].vy = vy[i];
    p[i].fx = fx[i];
    p[i].fy = fy[i];
}

__global__ void integrate_first_kernel(double *__restrict__ x,
                                       double *__restrict__ y,
                                       double *__restrict__ vx,
                                       double *__restrict__ vy,
                                       const double *__restrict__ fx,
                                       const double *__restrict__ fy,
                                       unsigned int n,
                                       double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double vxi = vx[i] + 0.5 * DT * fx[i];
    double vyi = vy[i] + 0.5 * DT * fy[i];
    double xi = x[i] + DT * vxi;
    double yi = y[i] + DT * vyi;

    if (xi >= box_size) xi -= box_size;
    else if (xi < 0.0) xi += box_size;
    if (yi >= box_size) yi -= box_size;
    else if (yi < 0.0) yi += box_size;

    x[i] = xi;
    y[i] = yi;
    vx[i] = vxi;
    vy[i] = vyi;
}

__global__ void integrate_second_kernel(double *__restrict__ vx,
                                        double *__restrict__ vy,
                                        const double *__restrict__ fx,
                                        const double *__restrict__ fy,
                                        unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
}

__global__ void clear_forces_kernel(double *__restrict__ fx,
                                    double *__restrict__ fy,
                                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] = 0.0;
    fy[i] = 0.0;
}

__global__ void clear_double_kernel(double *__restrict__ values, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    values[i] = 0.0;
}

__global__ void clear_cells_kernel(int *__restrict__ cell_counts,
                                   int *__restrict__ overflow,
                                   int total_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_cells) cell_counts[i] = 0;
    if (i == 0) *overflow = 0;
}

__global__ void build_cells_kernel(const double *__restrict__ x,
                                   const double *__restrict__ y,
                                   int *__restrict__ cell_counts,
                                   int *__restrict__ cell_particles,
                                   int *__restrict__ overflow,
                                   unsigned int n,
                                   int nc,
                                   double cell_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cx = (int)(x[i] / cell_size);
    int cy = (int)(y[i] / cell_size);

    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= nc) cx = nc - 1;
    if (cy >= nc) cy = nc - 1;

    int cell = cy * nc + cx;
    int slot = atomicAdd(&cell_counts[cell], 1);

    if (slot < MAX_PARTICLES_PER_CELL)
        cell_particles[cell * MAX_PARTICLES_PER_CELL + slot] = (int)i;
    else
        atomicExch(overflow, 1);
}

/*
 * Kernel za izračun sil s cell-list strukturo brez Newtonovega 3. zakona.
 *
 * Ena CUDA nit obdela en delec i:
 *   - Pregleda 3x3 = 9 sosednjih celic (z periodičnim ovojem).
 *   - Silo akumulira v lokalnih spremenljivkah fix, fiy.
 *   - Ob koncu zapiše silo samo v fx[i], fy[i] -> ni race conditiona, ni atomics.
 *   - PE upošteva faktor 0.5, ker vsak par (i,j) obiščemo dvakrat.
 */
__global__ void compute_forces_cell_kernel(const double *__restrict__ x,
                                           const double *__restrict__ y,
                                           double *__restrict__ fx,
                                           double *__restrict__ fy,
                                           double *__restrict__ pe_terms,
                                           const int *__restrict__ cell_counts,
                                           const int *__restrict__ cell_particles,
                                           unsigned int n,
                                           double box_size,
                                           int nc,
                                           double cell_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];
    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = lj_v_shift();

    double fix = 0.0, fiy = 0.0, local_pe = 0.0;

    int cx_i = (int)(xi / cell_size);
    int cy_i = (int)(yi / cell_size);
    if (cx_i < 0) cx_i = 0;
    if (cy_i < 0) cy_i = 0;
    if (cx_i >= nc) cx_i = nc - 1;
    if (cy_i >= nc) cy_i = nc - 1;

    for (int dcy = -1; dcy <= 1; dcy++)
    {
        for (int dcx = -1; dcx <= 1; dcx++)
        {
            int cx_j = wrap_cell(cx_i + dcx, nc);
            int cy_j = wrap_cell(cy_i + dcy, nc);
            int cell_j = cy_j * nc + cx_j;

            int count = cell_counts[cell_j];
            if (count > MAX_PARTICLES_PER_CELL) count = MAX_PARTICLES_PER_CELL;

            for (int k = 0; k < count; k++)
            {
                int j = cell_particles[cell_j * MAX_PARTICLES_PER_CELL + k];
                if ((unsigned int)j == i) continue;

                double dx = xi - x[j];
                double dy = yi - y[j];

                if (dx > half_box) dx -= box_size;
                else if (dx < -half_box) dx += box_size;
                if (dy > half_box) dy -= box_size;
                else if (dy < -half_box) dy += box_size;

                double r2 = dx * dx + dy * dy;
                if (r2 < rcut2 && r2 > 0.0)
                {
                    double inv_r2 = 1.0 / r2;
                    double sr2 = (SIGMA * SIGMA) * inv_r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                    fix += f_over_r * dx;
                    fiy += f_over_r * dy;
                    local_pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }
    }

    fx[i] = fix;
    fy[i] = fiy;
    pe_terms[i] = local_pe;
}

/*
 * All-pairs fallback za majhno število celic (nc < 3).
 * Ena nit obdela en delec i, zanko po vseh j != i.
 */
__global__ void compute_forces_allpairs_kernel(const double *__restrict__ x,
                                               const double *__restrict__ y,
                                               double *__restrict__ fx,
                                               double *__restrict__ fy,
                                               double *__restrict__ pe_terms,
                                               unsigned int n,
                                               double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];
    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = lj_v_shift();

    double fix = 0.0, fiy = 0.0, local_pe = 0.0;

    for (unsigned int j = 0; j < n; j++)
    {
        if (j == i) continue;

        double dx = xi - x[j];
        double dy = yi - y[j];

        if (dx > half_box) dx -= box_size;
        else if (dx < -half_box) dx += box_size;
        if (dy > half_box) dy -= box_size;
        else if (dy < -half_box) dy += box_size;

        double r2 = dx * dx + dy * dy;
        if (r2 < rcut2 && r2 > 0.0)
        {
            double inv_r2 = 1.0 / r2;
            double sr2 = (SIGMA * SIGMA) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
            fix += f_over_r * dx;
            fiy += f_over_r * dy;
            local_pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
        }
    }

    fx[i] = fix;
    fy[i] = fiy;
    pe_terms[i] = local_pe;
}

__global__ void kinetic_terms_kernel(const double *__restrict__ vx,
                                     const double *__restrict__ vy,
                                     double *__restrict__ terms,
                                     unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    terms[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
}

__global__ void reduce_sum_kernel(const double *__restrict__ in,
                                  double *__restrict__ out,
                                  unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    extern __shared__ double sdata[];
    double sum = 0.0;

    if (i < n) sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

static double reduce_device_sum(const double *d_values,
                                double *d_tmp1,
                                double *d_tmp2,
                                unsigned int n)
{
    if (n == 0) return 0.0;

    const double *in = d_values;
    double *out = d_tmp1;
    unsigned int cur_n = n;

    while (cur_n > 1)
    {
        unsigned int blocks = (cur_n + (REDUCE_THREADS * 2 - 1)) / (REDUCE_THREADS * 2);
        reduce_sum_kernel<<<blocks, REDUCE_THREADS, REDUCE_THREADS * sizeof(double)>>>(in, out, cur_n);
        CUDA_CHECK(cudaGetLastError());
        cur_n = blocks;
        in = out;
        out = (out == d_tmp1) ? d_tmp2 : d_tmp1;
    }

    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

static double compute_ke_gpu(const double *d_vx,
                             const double *d_vy,
                             double *d_terms,
                             double *d_tmp1,
                             double *d_tmp2,
                             unsigned int n)
{
    unsigned int blocks = (n + REDUCE_THREADS - 1) / REDUCE_THREADS;
    kinetic_terms_kernel<<<blocks, REDUCE_THREADS>>>(d_vx, d_vy, d_terms, n);
    CUDA_CHECK(cudaGetLastError());
    return reduce_device_sum(d_terms, d_tmp1, d_tmp2, n);
}

static void build_cells_gpu(const double *d_x,
                            const double *d_y,
                            int *d_cell_counts,
                            int *d_cell_particles,
                            int *d_overflow,
                            unsigned int n,
                            int nc,
                            double cell_size,
                            int total_cells)
{
    unsigned int cell_blocks = (total_cells + VECTOR_THREADS - 1) / VECTOR_THREADS;
    clear_cells_kernel<<<cell_blocks, VECTOR_THREADS>>>(d_cell_counts, d_overflow, total_cells);
    CUDA_CHECK(cudaGetLastError());

    unsigned int particle_blocks = (n + VECTOR_THREADS - 1) / VECTOR_THREADS;
    build_cells_kernel<<<particle_blocks, VECTOR_THREADS>>>(d_x, d_y, d_cell_counts, d_cell_particles,
                                                            d_overflow, n, nc, cell_size);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * Izračuna vse sile na GPE.
 * Pe_terms je vedno velikosti n (ena vrednost na delec).
 */
static void compute_forces_gpu(const double *d_x,
                               const double *d_y,
                               double *d_fx,
                               double *d_fy,
                               double *d_pe_terms,
                               int *d_cell_counts,
                               int *d_cell_particles,
                               int *d_overflow,
                               unsigned int n,
                               double box_size,
                               int nc,
                               double cell_size,
                               int total_cells,
                               int use_cells)
{
    unsigned int blocks = (n + VECTOR_THREADS - 1) / VECTOR_THREADS;

    if (use_cells)
    {
        build_cells_gpu(d_x, d_y, d_cell_counts, d_cell_particles, d_overflow,
                        n, nc, cell_size, total_cells);

#if CHECK_CELL_OVERFLOW
        int h_overflow = 0;
        CUDA_CHECK(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_overflow)
        {
            fprintf(stderr, "Cell list overflow. Increase MAX_PARTICLES_PER_CELL.\n");
            exit(EXIT_FAILURE);
        }
#endif

        compute_forces_cell_kernel<<<blocks, VECTOR_THREADS>>>(
            d_x, d_y, d_fx, d_fy, d_pe_terms,
            d_cell_counts, d_cell_particles, n, box_size, nc, cell_size);
        CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        compute_forces_allpairs_kernel<<<blocks, VECTOR_THREADS>>>(
            d_x, d_y, d_fx, d_fy, d_pe_terms, n, box_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -----------------------------------------------------------------------------
// Glavna CUDA simulacija
// -----------------------------------------------------------------------------

SimulationResult run_simulation(Particle *particles,
                                unsigned int n,
                                unsigned int nsteps,
                                double box_size,
                                int log_steps)
{
    SimulationResult out;
    memset(&out, 0, sizeof(out));
    out.n = n;
    out.particles = particles;

    if (n == 0)
        return out;

    const size_t particle_bytes = (size_t)n * sizeof(Particle);
    const size_t double_bytes = (size_t)n * sizeof(double);
    const unsigned int vector_blocks = (n + VECTOR_THREADS - 1) / VECTOR_THREADS;

    int nc = (int)(box_size / R_CUT);
    if (nc < 1) nc = 1;
    double cell_size = box_size / (double)nc;
    int total_cells = nc * nc;
    int use_cells = (nc >= 3);

    /* Pe_terms in ke_terms sta vedno velikosti n. */
    const unsigned int reduce_capacity = (n + (REDUCE_THREADS * 2 - 1)) / (REDUCE_THREADS * 2);
    const size_t reduce_bytes = (size_t)((reduce_capacity > 1) ? reduce_capacity : 1) * sizeof(double);

    Particle *d_particles = NULL;
    double *d_x = NULL, *d_y = NULL, *d_vx = NULL, *d_vy = NULL, *d_fx = NULL, *d_fy = NULL;
    double *d_pe_terms = NULL, *d_ke_terms = NULL, *d_tmp1 = NULL, *d_tmp2 = NULL;
    int *d_cell_counts = NULL, *d_cell_particles = NULL, *d_overflow = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_particles, particle_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vx, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vy, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fx, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fy, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_pe_terms, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ke_terms, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp1, reduce_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp2, reduce_bytes));

    if (use_cells)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_cell_counts, (size_t)total_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_cell_particles,
                              (size_t)total_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_overflow, sizeof(int)));
    }

    CUDA_CHECK(cudaMemcpy(d_particles, particles, particle_bytes, cudaMemcpyHostToDevice));
    pack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());

    compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_terms,
                       d_cell_counts, d_cell_particles, d_overflow,
                       n, box_size, nc, cell_size, total_cells, use_cells);
    out.start_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, n);
    out.start_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
    out.start_total = out.start_kinetic + out.start_potential;

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif)
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    else
    {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; ++step)
    {
        integrate_first_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_x, d_y, d_vx, d_vy,
                                                                  d_fx, d_fy, n, box_size);
        CUDA_CHECK(cudaGetLastError());

        compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_terms,
                           d_cell_counts, d_cell_particles, d_overflow,
                           n, box_size, nc, cell_size, total_cells, use_cells);

        integrate_second_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_vx, d_vy, d_fx, d_fy, n);
        CUDA_CHECK(cudaGetLastError());

        if (log_steps)
        {
            out.final_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, n);
            out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
            out.final_total = out.final_kinetic + out.final_potential;
            printf("step=%6u KE=%12.6f PE=%12.6f E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)
        {
            unpack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(particles, d_particles, particle_bytes, cudaMemcpyDeviceToHost));
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

    if (!log_steps && nsteps > 0)
    {
        out.final_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, n);
        out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
        out.final_total = out.final_kinetic + out.final_potential;
    }

    unpack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(particles, d_particles, particle_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

#if GENERATE_GIF
    if (gif)
        ge_close_gif(gif);
#endif

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_pe_terms));
    CUDA_CHECK(cudaFree(d_ke_terms));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));
    if (d_cell_counts) CUDA_CHECK(cudaFree(d_cell_counts));
    if (d_cell_particles) CUDA_CHECK(cudaFree(d_cell_particles));
    if (d_overflow) CUDA_CHECK(cudaFree(d_overflow));

    return out;
}
