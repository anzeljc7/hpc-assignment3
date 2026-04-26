#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "gifenc.h"
#include "lennard-jones.h"

#define CHECK_CUDA(call)                                                                               \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

// ---------------------------------------------------------------------------
// GIF Rendering (CPU)
// ---------------------------------------------------------------------------
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
        int py = (FRAME_HEIGHT - 1) - (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy)
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx)
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS)
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
    }
}
#endif

// ---------------------------------------------------------------------------
// Pomožne funkcije (CPU)
// ---------------------------------------------------------------------------
double random_double(void) { return (double)rand() / (double)RAND_MAX; }

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed, double temperature)
{
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;
    double mean_vx = 0.0, mean_vy = 0.0;

    for (unsigned int k = 0; k < n; k++)
    {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;
        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    for (unsigned int k = 0; k < n; k++)
    {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0)
        return 0;

    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++)
    {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// CUDA KERNELI (Matrika sil in Redukcija vrstic)
// ---------------------------------------------------------------------------

__global__ void leapfrog_step1_kernel(double *x, double *y, double *vx, double *vy, double *fx, double *fy, int n, double dt, double box_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        vx[i] += 0.5 * dt * fx[i];
        vy[i] += 0.5 * dt * fy[i];
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];

        double wx = fmod(x[i], box_size);
        double wy = fmod(y[i], box_size);
        if (wx < 0.0)
            wx += box_size;
        if (wy < 0.0)
            wy += box_size;
        x[i] = wx;
        y[i] = wy;
    }
}

// Inicializacija skalarjev (PE, KE)
__global__ void reset_scalars_kernel(double *global_pe, double *global_ke)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *global_pe = 0.0;
        *global_ke = 0.0;
    }
}

// KORAK 1: Matrika sil N x N (1 nit = 1 par i in j). Zagon 8000 x 8000 worka!
__global__ void compute_force_matrix_kernel(
    double *x, double *y, double *matrix_fx, double *matrix_fy,
    int n, int nc, double cell_size, double box_size, double half_box,
    double rc2, double v_shift, double *global_pe)
{
    // 2D Grid: x=column(j), y=row(i)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        // Newton 3: Gledamo le polovico matrike nad diagonalo
        if (j > i)
        {
            double fx_ij = 0.0;
            double fy_ij = 0.0;

            // Mreža (Grid) filter: Če sta celici predaleč, sploh ne računamo!
            int cx_i = (int)(x[i] / cell_size);
            int cy_i = (int)(y[i] / cell_size);
            int cx_j = (int)(x[j] / cell_size);
            int cy_j = (int)(y[j] / cell_size);

            if (cx_i >= nc)
                cx_i = nc - 1;
            if (cy_i >= nc)
                cy_i = nc - 1;
            if (cx_j >= nc)
                cx_j = nc - 1;
            if (cy_j >= nc)
                cy_j = nc - 1;

            // Periodična razdalja med celicami
            int dx_c = abs(cx_i - cx_j);
            int dy_c = abs(cy_i - cy_j);
            if (dx_c > nc / 2)
                dx_c = nc - dx_c;
            if (dy_c > nc / 2)
                dy_c = nc - dy_c;

            // Računamo fiziko SAMO, če sta celici sosednji
            if (dx_c <= 1 && dy_c <= 1)
            {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                if (dx > half_box)
                    dx -= box_size;
                else if (dx < -half_box)
                    dx += box_size;
                if (dy > half_box)
                    dy -= box_size;
                else if (dy < -half_box)
                    dy += box_size;

                double r2 = dx * dx + dy * dy;
                if (r2 < rc2 && r2 > 0.0)
                {
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;

                    fx_ij = fij_r2 * dx;
                    fy_ij = fij_r2 * dy;

                    double pe_ij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
                    atomicAdd(global_pe, pe_ij);
                }
            }

            // Varno pisanje v globalno matriko brez atomic (akcija in reakcija)
            matrix_fx[i * n + j] = fx_ij;
            matrix_fy[i * n + j] = fy_ij;
            matrix_fx[j * n + i] = -fx_ij;
            matrix_fy[j * n + i] = -fy_ij;
        }
        else if (i == j)
        {
            // Diagonala je 0
            matrix_fx[i * n + i] = 0.0;
            matrix_fy[i * n + i] = 0.0;
        }
    }
}

// KORAK 2: "Blok reduca row" - 1 Blok = 1 Vrstica matrike (z 1024 nitmi)
__global__ void reduce_row_kernel(double *matrix_fx, double *matrix_fy, double *fx, double *fy, int n)
{
    int row = blockIdx.x; // Točno en blok za vsakega delca
    if (row >= n)
        return;

    int tid = threadIdx.x;
    double sum_fx = 0.0;
    double sum_fy = 0.0;

    // Niti v bloku hkrati posesa celotno vrstico (vseh N stolpcev)
    for (int col = tid; col < n; col += blockDim.x)
    {
        sum_fx += matrix_fx[row * n + col];
        sum_fy += matrix_fy[row * n + col];
    }

    extern __shared__ double s_data[];
    double *s_fx = s_data;
    double *s_fy = &s_data[blockDim.x];

    s_fx[tid] = sum_fx;
    s_fy[tid] = sum_fy;
    __syncthreads();

    // Redukcija v skupnem pomnilniku (Shared Memory)
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_fx[tid] += s_fx[tid + s];
            s_fy[tid] += s_fy[tid + s];
        }
        __syncthreads();
    }

    // Rezultat končne sile za delca se prepiše iz bloka
    if (tid == 0)
    {
        fx[row] = s_fx[0];
        fy[row] = s_fy[0];
    }
}

__global__ void step2_kernel(double *vx, double *vy, double *fx, double *fy, int n, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        vx[i] += 0.5 * dt * fx[i];
        vy[i] += 0.5 * dt * fy[i];
    }
}

__global__ void ke_kernel(double *vx, double *vy, int n, double *global_ke)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double ke_local = (i < n) ? 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]) : 0.0;

    extern __shared__ double s_ke[];
    int tid = threadIdx.x;
    s_ke[tid] = ke_local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            s_ke[tid] += s_ke[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(global_ke, s_ke[0]);
}

// ---------------------------------------------------------------------------
// Main Zanka
// ---------------------------------------------------------------------------

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps)
{
    SimulationResult out;

    int threads1D = 1024;
    int blocks1D = (n + threads1D - 1) / threads1D;

    double v_shift_host = 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12) - pow(SIGMA / R_CUT, 6));

    int nc = (int)(box_size / R_CUT);
    if (nc < 1)
        nc = 1;
    double cell_size = box_size / (double)nc;

    // SoA + Matrika sil Alokacija
    double *d_x, *d_y, *d_vx, *d_vy, *d_fx, *d_fy, *d_pe, *d_ke;
    double *d_matrix_fx, *d_matrix_fy;

    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vx, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vy, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fx, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fy, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pe, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_ke, sizeof(double)));

    // Ogromni matriki za izogib atomičnim operacijam na silah
    CHECK_CUDA(cudaMalloc(&d_matrix_fx, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_matrix_fy, n * n * sizeof(double)));

    double *h_x = (double *)malloc(n * sizeof(double));
    double *h_y = (double *)malloc(n * sizeof(double));
    double *h_vx = (double *)malloc(n * sizeof(double));
    double *h_vy = (double *)malloc(n * sizeof(double));

    for (unsigned int i = 0; i < n; i++)
    {
        h_x[i] = particles[i].x;
        h_y[i] = particles[i].y;
        h_vx[i] = particles[i].vx;
        h_vy[i] = particles[i].vy;
    }

    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vx, h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vy, h_vy, n * sizeof(double), cudaMemcpyHostToDevice));

    // Parametri za 2D mrežo sil (NxN work)
    dim3 threads2D(32, 32);
    dim3 blocks2D((n + threads2D.x - 1) / threads2D.x, (n + threads2D.y - 1) / threads2D.y);

    // Prvi krog izračunov
    reset_scalars_kernel<<<1, 1>>>(d_pe, d_ke);
    compute_force_matrix_kernel<<<blocks2D, threads2D>>>(d_x, d_y, d_matrix_fx, d_matrix_fy, n, nc, cell_size, box_size, 0.5 * box_size, R_CUT * R_CUT, v_shift_host, d_pe);

    // "Blok reduca row" z 1024 nitmi
    int sharedMemRow = 2 * threads1D * sizeof(double);
    reduce_row_kernel<<<n, threads1D, sharedMemRow>>>(d_matrix_fx, d_matrix_fy, d_fx, d_fy, n);

    ke_kernel<<<blocks1D, threads1D, threads1D * sizeof(double)>>>(d_vx, d_vy, n, d_ke);

    CHECK_CUDA(cudaMemcpy(&out.start_potential, d_pe, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&out.start_kinetic, d_ke, sizeof(double), cudaMemcpyDeviceToHost));
    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (gif)
    {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    // GLAVNA ZANKA (Izvaja se izključno na GPU)
    for (unsigned int step = 0; step < nsteps; step++)
    {
        leapfrog_step1_kernel<<<blocks1D, threads1D>>>(d_x, d_y, d_vx, d_vy, d_fx, d_fy, n, DT, box_size);

        reset_scalars_kernel<<<1, 1>>>(d_pe, d_ke);

        // Polnjenje matrike sil (i * j work) z Grid preskakovanjem
        compute_force_matrix_kernel<<<blocks2D, threads2D>>>(d_x, d_y, d_matrix_fx, d_matrix_fy, n, nc, cell_size, box_size, 0.5 * box_size, R_CUT * R_CUT, v_shift_host, d_pe);

        // Redukcija posamezne vrstice (1 Blok = 1 Delec)
        reduce_row_kernel<<<n, threads1D, sharedMemRow>>>(d_matrix_fx, d_matrix_fy, d_fx, d_fy, n);

        step2_kernel<<<blocks1D, threads1D>>>(d_vx, d_vy, d_fx, d_fy, n, DT);

        // POPRAVEK TUKAJ: Varno preverjanje za kopiranje na CPU (rešena napaka z undefined "gif")
        int copy_data = 0;
        if (log_steps)
            copy_data = 1;
        if (step == nsteps - 1)
            copy_data = 1;
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)
            copy_data = 1;
#endif

        if (copy_data)
        {
            ke_kernel<<<blocks1D, threads1D, threads1D * sizeof(double)>>>(d_vx, d_vy, n, d_ke);

            CHECK_CUDA(cudaMemcpy(&out.final_potential, d_pe, sizeof(double), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&out.final_kinetic, d_ke, sizeof(double), cudaMemcpyDeviceToHost));
            out.final_total = out.final_kinetic + out.final_potential;

            if (log_steps)
                printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n", step, out.final_kinetic, out.final_potential, out.final_total);

#if GENERATE_GIF
            if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)
            {
                CHECK_CUDA(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
                for (unsigned int i = 0; i < n; i++)
                {
                    particles[i].x = h_x[i];
                    particles[i].y = h_y[i];
                }
                render_frame_gif(gif, particles, n, box_size);
                ge_add_frame(gif, FRAME_DELAY);
            }
#endif
        }
    }

#if GENERATE_GIF
    if (gif)
        ge_close_gif(gif);
#endif

    // Nazaj v AoS
    CHECK_CUDA(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vx, d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vy, d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < n; i++)
    {
        particles[i].x = h_x[i];
        particles[i].y = h_y[i];
        particles[i].vx = h_vx[i];
        particles[i].vy = h_vy[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_pe);
    cudaFree(d_ke);
    cudaFree(d_matrix_fx);
    cudaFree(d_matrix_fy);
    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);

    out.n = n;
    out.particles = particles;
    return out;
}