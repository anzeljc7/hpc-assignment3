#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// #include <cuda_runtime.h>
// #include <cuda.h>

#include "gifenc.h"
#include "lennard-jones.h"

// ---------------------------------------------------------------------------
// GIF rendering
// ---------------------------------------------------------------------------
#if GENERATE_GIF
uint8_t palette[] = {0, 0, 0, 255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index)
{
    if (x < 0 || y < 0 || x >= w || y >= h)
        return;
    img[(size_t)y * (size_t)w + (size_t)x] = index;
}

void render_frame_gif(ge_GIF *gif, const Particle *particles,
                      unsigned int n, double box_size)
{
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);
    for (unsigned int i = 0; i < n; ++i)
    {
        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (FRAME_HEIGHT - 1) - (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy)
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx)
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS)
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT,
                              px + dx, py + dy, 1);
    }
}
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

double random_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

double compute_ke(const Particle *particles, unsigned int n)
{
    double ke = 0.0;
#pragma omp parallel for reduction(+ : ke) schedule(static)
    for (unsigned int i = 0; i < n; ++i)
        ke += 0.5 * (particles[i].vx * particles[i].vx +
                     particles[i].vy * particles[i].vy);
    return ke;
}

// ---------------------------------------------------------------------------
// Particle initialisation
// ---------------------------------------------------------------------------

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature)
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
        ke += 0.5 * (particles[k].vx * particles[k].vx +
                     particles[k].vy * particles[k].vy);
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
// Periodic boundary conditions
// ---------------------------------------------------------------------------

void wrap_positions(Particle *particles, unsigned int n, double box_size)
{
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        double wx = fmod(particles[i].x, box_size);
        double wy = fmod(particles[i].y, box_size);
        if (wx < 0.0)
            wx += box_size;
        if (wy < 0.0)
            wy += box_size;
        particles[i].x = wx;
        particles[i].y = wy;
    }
}

// ---------------------------------------------------------------------------
// Shifted LJ potential
// ---------------------------------------------------------------------------

double compute_v_shift(void)
{
    double sr_cut = SIGMA / R_CUT;
    double sr6 = sr_cut * sr_cut * sr_cut * sr_cut * sr_cut * sr_cut;
    return 4.0 * EPSILON * (sr6 * sr6 - sr6);
}

// ---------------------------------------------------------------------------
// Cell list
// ---------------------------------------------------------------------------

typedef struct
{
    int *head;
    int *next;
    int nc;
    double cell_size;
    int n_cells;
} CellList;

static CellList celllist_create(unsigned int n, double box_size)
{
    CellList cl;
    cl.nc = (int)(box_size / R_CUT);
    if (cl.nc < 1)
        cl.nc = 1;
    cl.cell_size = box_size / (double)cl.nc;
    cl.n_cells = cl.nc * cl.nc;
    cl.head = (int *)malloc((size_t)cl.n_cells * sizeof(int));
    cl.next = (int *)malloc((size_t)n * sizeof(int));
    return cl;
}

static void celllist_free(CellList *cl)
{
    free(cl->head);
    free(cl->next);
}

static void celllist_build(CellList *cl, const Particle *particles,
                           unsigned int n)
{
    for (int c = 0; c < cl->n_cells; c++)
        cl->head[c] = -1;
    for (unsigned int i = 0; i < n; i++)
    {
        int cx = (int)(particles[i].x / cl->cell_size);
        int cy = (int)(particles[i].y / cl->cell_size);
        if (cx >= cl->nc)
            cx = cl->nc - 1;
        if (cy >= cl->nc)
            cy = cl->nc - 1;
        int c = cy * cl->nc + cx;
        cl->next[i] = cl->head[c];
        cl->head[c] = (int)i;
    }
}

// ---------------------------------------------------------------------------
// Force computation: HALF-SHELL + CELL LISTS (N*(N-1)/2 operacij)
// ---------------------------------------------------------------------------

double compute_forces(Particle *particles, unsigned int n, double box_size, CellList *cl)
{
    const double v_shift = compute_v_shift();
    const double rc2 = R_CUT * R_CUT;
    const double half_box = 0.5 * box_size;

    celllist_build(cl, particles, n);

    // Vse sile nastavimo na 0, preden začnemo prištevati in odštevati
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    double pe = 0.0;

    // 5 smeri iskanja za prepolovitev: lastna celica, desno, levo-spodaj, spodaj, desno-spodaj
    int offsets[5][2] = {{0, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

#pragma omp parallel for reduction(+ : pe) schedule(dynamic, 16)
    for (unsigned int i = 0; i < n; ++i)
    {
        int cx_i = (int)(particles[i].x / cl->cell_size);
        int cy_i = (int)(particles[i].y / cl->cell_size);
        if (cx_i >= cl->nc)
            cx_i = cl->nc - 1;
        if (cy_i >= cl->nc)
            cy_i = cl->nc - 1;

        for (int dir = 0; dir < 5; ++dir)
        {
            int dcx = offsets[dir][0];
            int dcy = offsets[dir][1];

            int cx_j = ((cx_i + dcx) % cl->nc + cl->nc) % cl->nc;
            int cy_j = ((cy_i + dcy) % cl->nc + cl->nc) % cl->nc;
            int cell_j = cy_j * cl->nc + cx_j;

            for (int j = cl->head[cell_j]; j != -1; j = cl->next[j])
            {
                // Preprečimo dvojno računanje: v ISTI celici ignoriramo delce z manjšo ali enako id
                if (dir == 0 && (unsigned int)j <= i)
                    continue;

                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;

                if (dx > half_box)
                    dx -= box_size;
                else if (dx < -half_box)
                    dx += box_size;
                if (dy > half_box)
                    dy -= box_size;
                else if (dy < -half_box)
                    dy += box_size;

                double r2 = dx * dx + dy * dy;
                if (r2 >= rc2 || r2 == 0.0)
                    continue;

                double sr2 = (SIGMA * SIGMA) / r2;
                double sr6 = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;

                double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                double fx = fij_r2 * dx;
                double fy = fij_r2 * dy;

                // ATOMIC ZAPISOVANJE (Newtonov 3. zakon)
#pragma omp atomic
                particles[i].fx += fx;
#pragma omp atomic
                particles[i].fy += fy;

#pragma omp atomic
                particles[j].fx -= fx;
#pragma omp atomic
                particles[j].fy -= fy;

                // Ne delimo več z 0.5, ker par (i, j) obiščemo samo enkrat
                pe += (4.0 * EPSILON * (sr12 - sr6) - v_shift);
            }
        }
    }

    return pe;
}

// ---------------------------------------------------------------------------
// Leapfrog integrator
// ---------------------------------------------------------------------------

double leapfrog_step(Particle *particles, unsigned int n, double box_size, CellList *cl)
{
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);

    double pe = compute_forces(particles, n, box_size, cl);

#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

// ---------------------------------------------------------------------------
// Top-level simulation runner
// ---------------------------------------------------------------------------

SimulationResult run_simulation(Particle *particles, unsigned int n,
                                unsigned int nsteps, double box_size,
                                int log_steps)
{
    SimulationResult out;

    CellList cl = celllist_create(n, box_size);

    out.start_potential = compute_forces(particles, n, box_size, &cl);
    out.start_kinetic = compute_ke(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE,
                             (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT,
                             palette, 8, -1, 0);
    if (!gif)
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    else
    {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++)
    {
        out.final_potential = leapfrog_step(particles, n, box_size, &cl);
        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;

        if (log_steps)
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)
        {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif)
        ge_close_gif(gif);
#endif

    celllist_free(&cl);

    out.n = n;
    out.particles = particles;
    return out;
}