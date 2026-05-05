#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gifenc.h"
#include "lennard-jones.h"

#ifndef FORCE_THREADS
#define FORCE_THREADS 128
#endif

#ifndef REDUCE_THREADS
#define REDUCE_THREADS 128
#endif

#define CUDA_CHECK(call)                                                         \
    do {                                                                        \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_err));                                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)

#if GENERATE_GIF
uint8_t palette[] = {0, 0, 0, 255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}

void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size) {
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);
    for (unsigned int i = 0; i < n; ++i) {
        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;
        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif

double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

/* Ni uporabljena v GPU poti — run_simulation uporablja compute_ke_gpu(). */
double compute_ke(const Particle *particles, unsigned int n) {
    (void)particles; (void)n;
    return 0.0;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature) {
    srand(seed);

    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;

    for (unsigned int k = 0; k < n; k++) {
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
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

/* Ni uporabljena v GPU poti — integrate_first_kernel zavija pozicije inline. */
void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    (void)particles; (void)n; (void)box_size;
}

static __host__ __device__ inline double lj_v_shift(void) {
    const double sr = SIGMA / R_CUT;
    const double sr2 = sr * sr;
    const double sr6 = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;
    return 4.0 * EPSILON * (sr12 - sr6);
}

double compute_v_shift(void) {
    return lj_v_shift();
}

/* Ni uporabljena v GPU poti — run_simulation uporablja compute_forces_kernel(). */
double compute_forces(Particle *particles, unsigned int n, double box_size) {
    (void)particles; (void)n; (void)box_size;
    return 0.0;
}

/* Ni uporabljena v GPU poti — run_simulation uporablja integrate_first/second_kernel. */
double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    (void)particles; (void)n; (void)box_size;
    return 0.0;
}

__global__ void pack_particles_kernel(const Particle *__restrict__ p,
                                      double *__restrict__ x,
                                      double *__restrict__ y,
                                      double *__restrict__ vx,
                                      double *__restrict__ vy,
                                      double *__restrict__ fx,
                                      double *__restrict__ fy,
                                      unsigned int n) {
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
                                        unsigned int n) {
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
                                       double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double vxi = vx[i] + 0.5 * DT * fx[i];
    double vyi = vy[i] + 0.5 * DT * fy[i];
    double xi = x[i] + DT * vxi;
    double yi = y[i] + DT * vyi;

    // Periodic boundary condition. Works for both positive and negative values.
    xi -= box_size * floor(xi / box_size);
    yi -= box_size * floor(yi / box_size);

    x[i] = xi;
    y[i] = yi;
    vx[i] = vxi;
    vy[i] = vyi;
}

__global__ void integrate_second_kernel(double *__restrict__ vx,
                                        double *__restrict__ vy,
                                        const double *__restrict__ fx,
                                        const double *__restrict__ fy,
                                        unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
}

// One CUDA block computes one particle row. Threads in the block split the j-loop
// and then reduce force and potential energy for that particle.
__global__ __launch_bounds__(FORCE_THREADS)
void compute_forces_kernel(const double *__restrict__ x,
                           const double *__restrict__ y,
                           double *__restrict__ fx,
                           double *__restrict__ fy,
                           double *__restrict__ pe_rows,
                           unsigned int n,
                           double box_size) {
    unsigned int i = blockIdx.x;
    unsigned int tid = threadIdx.x;

    extern __shared__ double smem[];
    double *sfx = smem;
    double *sfy = sfx + blockDim.x;
    double *spe = sfy + blockDim.x;

    double xi = x[i];
    double yi = y[i];
    double inv_box = 1.0 / box_size;
    double rcut2 = R_CUT * R_CUT;
    double v_shift = lj_v_shift();

    double local_fx = 0.0;
    double local_fy = 0.0;
    double local_pe = 0.0;

    for (unsigned int j = tid; j < n; j += blockDim.x) {
        if (j == i) continue;

        double dx = xi - x[j];
        double dy = yi - y[j];

        dx -= box_size * nearbyint(dx * inv_box);
        dy -= box_size * nearbyint(dy * inv_box);

        double r2 = dx * dx + dy * dy;
        if (r2 < rcut2 && r2 > 0.0) {
            double inv_r2 = 1.0 / r2;
            double sr2 = (SIGMA * SIGMA) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
            local_fx += f_over_r * dx;
            local_fy += f_over_r * dy;
            local_pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
        }
    }

    sfx[tid] = local_fx;
    sfy[tid] = local_fy;
    spe[tid] = local_pe;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sfx[tid] += sfx[tid + stride];
            sfy[tid] += sfy[tid + stride];
            spe[tid] += spe[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        fx[i] = sfx[0];
        fy[i] = sfy[0];
        // Each pair is visited twice: i->j and j->i. Store half per row.
        pe_rows[i] = 0.5 * spe[0];
    }
}

__global__ void kinetic_terms_kernel(const double *__restrict__ vx,
                                     const double *__restrict__ vy,
                                     double *__restrict__ terms,
                                     unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    terms[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
}

__global__ void reduce_sum_kernel(const double *__restrict__ in,
                                  double *__restrict__ out,
                                  unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    extern __shared__ double sdata[];
    double sum = 0.0;

    if (i < n) sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

static double reduce_device_sum(const double *d_values,
                                double *d_tmp1,
                                double *d_tmp2,
                                unsigned int n) {
    if (n == 0) return 0.0;

    const double *in = d_values;
    double *out = d_tmp1;
    unsigned int cur_n = n;
    bool using_tmp1 = false;

    while (cur_n > 1) {
        unsigned int blocks = (cur_n + (REDUCE_THREADS * 2 - 1)) / (REDUCE_THREADS * 2);
        reduce_sum_kernel<<<blocks, REDUCE_THREADS, REDUCE_THREADS * sizeof(double)>>>(in, out, cur_n);
        CUDA_CHECK(cudaGetLastError());

        cur_n = blocks;
        in = out;
        using_tmp1 = (out == d_tmp1);
        out = using_tmp1 ? d_tmp2 : d_tmp1;
    }

    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

static void compute_forces_gpu(const double *d_x,
                               const double *d_y,
                               double *d_fx,
                               double *d_fy,
                               double *d_pe_rows,
                               unsigned int n,
                               double box_size) {
    size_t shmem = 3 * FORCE_THREADS * sizeof(double);
    compute_forces_kernel<<<n, FORCE_THREADS, shmem>>>(d_x, d_y, d_fx, d_fy, d_pe_rows, n, box_size);
    CUDA_CHECK(cudaGetLastError());
}

static double compute_ke_gpu(const double *d_vx,
                             const double *d_vy,
                             double *d_terms,
                             double *d_tmp1,
                             double *d_tmp2,
                             unsigned int n) {
    unsigned int blocks = (n + REDUCE_THREADS - 1) / REDUCE_THREADS;
    kinetic_terms_kernel<<<blocks, REDUCE_THREADS>>>(d_vx, d_vy, d_terms, n);
    CUDA_CHECK(cudaGetLastError());
    return reduce_device_sum(d_terms, d_tmp1, d_tmp2, n);
}

SimulationResult run_simulation(Particle *particles,
                                unsigned int n,
                                unsigned int nsteps,
                                double box_size,
                                int log_steps) {
    SimulationResult out;
    memset(&out, 0, sizeof(out));
    out.n = n;
    out.particles = particles;

    if (n == 0) {
        return out;
    }

    const size_t bytes = (size_t)n * sizeof(double);
    const unsigned int vector_blocks = (n + REDUCE_THREADS - 1) / REDUCE_THREADS;
    const unsigned int reduce_capacity = (n + (REDUCE_THREADS * 2 - 1)) / (REDUCE_THREADS * 2);
    const size_t reduce_bytes = (size_t)((reduce_capacity > 1) ? reduce_capacity : 1) * sizeof(double);

    Particle *d_particles = NULL;
    double *d_x = NULL, *d_y = NULL, *d_vx = NULL, *d_vy = NULL, *d_fx = NULL, *d_fy = NULL;
    double *d_pe_rows = NULL, *d_ke_terms = NULL, *d_tmp1 = NULL, *d_tmp2 = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_particles, (size_t)n * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc((void **)&d_x, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vx, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vy, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fx, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fy, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_pe_rows, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ke_terms, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp1, reduce_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp2, reduce_bytes));

    CUDA_CHECK(cudaMemcpy(d_particles, particles, (size_t)n * sizeof(Particle), cudaMemcpyHostToDevice));
    pack_particles_kernel<<<vector_blocks, REDUCE_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());

    compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_rows, n, box_size);
    out.start_potential = reduce_device_sum(d_pe_rows, d_tmp1, d_tmp2, n);
    out.start_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = NULL;
    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        CUDA_CHECK(cudaMemcpy(particles, d_particles, (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

    for (unsigned int step = 0; step < nsteps; ++step) {
        integrate_first_kernel<<<vector_blocks, REDUCE_THREADS>>>(d_x, d_y, d_vx, d_vy, d_fx, d_fy, n, box_size);
        CUDA_CHECK(cudaGetLastError());

        compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_rows, n, box_size);

        integrate_second_kernel<<<vector_blocks, REDUCE_THREADS>>>(d_vx, d_vy, d_fx, d_fy, n);
        CUDA_CHECK(cudaGetLastError());

        if (log_steps) {
            out.final_potential = reduce_device_sum(d_pe_rows, d_tmp1, d_tmp2, n);
            out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
            out.final_total = out.final_kinetic + out.final_potential;
            printf("step=%6u KE=%12.6f PE=%12.6f E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            unpack_particles_kernel<<<vector_blocks, REDUCE_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(particles, d_particles, (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

    if (!log_steps && nsteps > 0) {
        out.final_potential = reduce_device_sum(d_pe_rows, d_tmp1, d_tmp2, n);
        out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
        out.final_total = out.final_kinetic + out.final_potential;
    }

    unpack_particles_kernel<<<vector_blocks, REDUCE_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(particles, d_particles, (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_pe_rows));
    CUDA_CHECK(cudaFree(d_ke_terms));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    return out;
}
