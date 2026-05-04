#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "gifenc.h"
#include "lennard-jones.h"

/*
 * CPU implementacija z dvema dodatnima optimizacijama glede na lennard-jones-optimized.cu:
 *
 * 1) Newtonov 3. zakon
 *    Vsak par (i, j) izračunamo samo enkrat (j > i).
 *    Delcu i prištejemo +F, delcu j pa -F.
 *    Ker je zunanji for loop vzporeden, več niti hkrati piše v fx[j] istega delca
 *    → race condition → rešujemo z #pragma omp atomic (CPU analog GPU-jevega atomicAdd).
 *
 * 2) Cell list (nespremenjen iz lennard-jones-optimized.cu)
 *    Zmanjša število pregledanih parov iz O(N²) na O(N).
 *
 * Namen te datoteke:
 *    Preveriti, ali atomic operacije na CPU (omp atomic) dejansko pomagajo
 *    ali pa overhead atomics pokvari pridobitev iz prepolovitve dela.
 *    Na GPU atomicAdd povzroča serializacijo pri konfliktih – na CPU je
 *    situacija drugačna (cache coherency protokol), a atomic prav tako ni zastonj.
 */

// ---------------------------------------------------------------------------
// Helpers (enaki kot v lennard-jones-optimized.cu)
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
// Particle initialisation (nespremenjena)
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
        if (wx < 0.0) wx += box_size;
        if (wy < 0.0) wy += box_size;
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
// Cell list (nespremenjen iz lennard-jones-optimized.cu)
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
    if (cl.nc < 1) cl.nc = 1;
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

static void celllist_build(CellList *cl, const Particle *particles, unsigned int n)
{
    for (int c = 0; c < cl->n_cells; c++)
        cl->head[c] = -1;
    for (unsigned int i = 0; i < n; i++)
    {
        int cx = (int)(particles[i].x / cl->cell_size);
        int cy = (int)(particles[i].y / cl->cell_size);
        if (cx >= cl->nc) cx = cl->nc - 1;
        if (cy >= cl->nc) cy = cl->nc - 1;
        int c = cy * cl->nc + cx;
        cl->next[i] = cl->head[c];
        cl->head[c] = (int)i;
    }
}

// ---------------------------------------------------------------------------
// Force computation – Newton 3. zakon + omp atomic
//
// Vzorec "ena nit, en delec i":
//   - nit i pregleduje samo j > i (Newton: vsak par enkrat)
//   - silo +F prištejem delcu i v lokalne spremenljivke fix/fiy (brez konflikta)
//   - silo -F prištejem delcu j z #pragma omp atomic (analog GPU atomicAdd)
//
// Kompromis:
//   + ~2x manj parov kot v lennard-jones-optimized.cu
//   - atomic operacije imajo overhead (cache invalidation med jedri)
//   - pri gostih sistemih je konfliktov več → večji overhead
//
// Za primerjavo z lennard-jones-optimized.cu:
//   Če je cpu_newton hitrejši: atomic overhead < pridobitev iz Newtona
//   Če je cpu_newton počasnejši: atomic overhead > pridobitev iz Newtona
// ---------------------------------------------------------------------------

double compute_forces(Particle *particles, unsigned int n, double box_size)
{
    const double v_shift = compute_v_shift();
    const double rc2 = R_CUT * R_CUT;
    const double half_box = 0.5 * box_size;

    // Ponastavi sile (vzporedno)
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    CellList cl = celllist_create(n, box_size);
    celllist_build(&cl, particles, n);

    double pe = 0.0;

#pragma omp parallel for reduction(+ : pe) schedule(dynamic, 16)
    for (unsigned int i = 0; i < n; ++i)
    {
        // Lokalni akumulatorji za delec i – samo ta nit piše sem → brez konflikta
        double fix = 0.0, fiy = 0.0;

        int cx_i = (int)(particles[i].x / cl.cell_size);
        int cy_i = (int)(particles[i].y / cl.cell_size);
        if (cx_i >= cl.nc) cx_i = cl.nc - 1;
        if (cy_i >= cl.nc) cy_i = cl.nc - 1;

        for (int dcy = -1; dcy <= 1; dcy++)
        {
            for (int dcx = -1; dcx <= 1; dcx++)
            {
                int cx_j = ((cx_i + dcx) % cl.nc + cl.nc) % cl.nc;
                int cy_j = ((cy_i + dcy) % cl.nc + cl.nc) % cl.nc;
                int cell_j = cy_j * cl.nc + cx_j;

                for (int j = cl.head[cell_j]; j != -1; j = cl.next[j])
                {
                    // Newton: obdelamo samo j > i → vsak par natanko enkrat
                    if ((unsigned int)j <= i)
                        continue;

                    double dx = particles[i].x - particles[j].x;
                    double dy = particles[i].y - particles[j].y;
                    if (dx > half_box)  dx -= box_size;
                    else if (dx < -half_box) dx += box_size;
                    if (dy > half_box)  dy -= box_size;
                    else if (dy < -half_box) dy += box_size;

                    double r2 = dx * dx + dy * dy;
                    if (r2 >= rc2 || r2 == 0.0)
                        continue;

                    double sr2  = (SIGMA * SIGMA) / r2;
                    double sr6  = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;

                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    double fij_x = fij_r2 * dx;
                    double fij_y = fij_r2 * dy;

                    // Delec i: lokalni akumulator – ta nit je edina, ki ga bere/piše
                    fix += fij_x;
                    fiy += fij_y;

                    // Delec j: več niti lahko hkrati piše sem → omp atomic (analog atomicAdd)
                    // Vsaka atomic operacija zaklene pomnilniško lokacijo za ostala jedra.
                    // Pri gostih sistemih je to ozko grlo.
#pragma omp atomic
                    particles[j].fx -= fij_x;
#pragma omp atomic
                    particles[j].fy -= fij_y;

                    // Par je štet enkrat (ne 0.5 faktor kot v lennard-jones-optimized.cu)
                    pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
                }
            }
        }

        // Delec i posodablja samo ta nit → navaden vpis, brez atomic
        particles[i].fx += fix;
        particles[i].fy += fiy;
    }

    celllist_free(&cl);
    return pe;
}

// ---------------------------------------------------------------------------
// Leapfrog integrator (nespremenjen)
// ---------------------------------------------------------------------------

double leapfrog_step(Particle *particles, unsigned int n, double box_size)
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
    double pe = compute_forces(particles, n, box_size);

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

    out.start_potential = compute_forces(particles, n, box_size);
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
        out.final_potential = leapfrog_step(particles, n, box_size);
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

    out.n = n;
    out.particles = particles;
    return out;
}
