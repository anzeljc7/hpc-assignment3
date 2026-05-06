#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "gifenc.h"
#include "lennard-jones.h"

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

double random_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

// Kinetična energija
// reduction(+:ke) zagotavlja, da vsaka nit sešteva v svojo spremenljivko ke,
// na koncu se sešteje brez race conditiona
double compute_ke(const Particle *particles, unsigned int n)
{
    double ke = 0.0;
#pragma omp parallel for reduction(+ : ke) schedule(static)
    for (unsigned int i = 0; i < n; ++i)
        ke += 0.5 * (particles[i].vx * particles[i].vx +
                     particles[i].vy * particles[i].vy);
    return ke;
}

// Particle initialisation

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature)
{
    srand(seed);

    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    // placement_fraction < 1: delci se postavijo v srednji del škatle,
    // da se izognemo takojšnjim trkom z robom pri visokih temperaturah.
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

    // Remove centre-of-mass drift → zero net momentum
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

    // Rescale velocities to match target temperature: Ek = N*T
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++)
    {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }
    return 1;
}

// Periodic boundary conditions
// Periodični robni pogoji: delci, ki zapustijo škatlo, se pojavijo na nasprotni strani.
void wrap_positions(Particle *particles, unsigned int n, double box_size)
{
// Spet neodvisni delci
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        // ostanek pri deljenju poskrbi, da je znotraj škatle
        double wx = fmod(particles[i].x, box_size);
        double wy = fmod(particles[i].y, box_size);
        // fmod vrne vrednost v [-box_size, box_size]
        // negativne se prestavijo v pozitivne
        if (wx < 0.0)
            wx += box_size;
        if (wy < 0.0)
            wy += box_size;
        particles[i].x = wx;
        particles[i].y = wy;
    }
}

// Shifted LJ potential
double compute_v_shift(void)
{
    double sr_cut = SIGMA / R_CUT;
    double sr6 = sr_cut * sr_cut * sr_cut * sr_cut * sr_cut * sr_cut;
    return 4.0 * EPSILON * (sr6 * sr6 - sr6);
}

// Cell list
// Ker delci zunaj R_CUT ne vplivajo drug na drugega, vsak delec i
// pregleda samo 3×3 = 9 sosednjih celic namesto vseh N delcev.
// Cell list se obnovi vsak korak, ker se delci premikajo.

typedef struct
{
    int *head; // indeks prvega delca v celici c, -1 če prazna
    int *next; // naslednji delec v isti celici, -1 če zadnji
    int nc;    // število celic na
    double cell_size;
    int n_cells;
} CellList;

static CellList celllist_create(unsigned int n, double box_size)
{
    CellList cl;

    // 1. Število celic na stran
    // Zaokroži na dol št. celic
    cl.nc = (int)(box_size / R_CUT);

    // 2. Če bi bila škatla manjša od R_CUT, uporabimo eno samo celico.
    if (cl.nc < 1)
        cl.nc = 1;

    // 3. Dejanska velikost celice — ker se zaokroži navzdol jšt celic je lahko velikost malo večja
    cl.cell_size = box_size / (double)cl.nc;
    cl.n_cells = cl.nc * cl.nc;

    // 4. Alokacija za delce
    cl.head = (int *)malloc((size_t)cl.n_cells * sizeof(int));
    cl.next = (int *)malloc((size_t)n * sizeof(int));
    return cl;
}

static void celllist_free(CellList *cl)
{
    free(cl->head);
    free(cl->next);
}

// Gradi povezan seznam delcev po celicah z vstavljanjem na začetek
// head[c] kaže na prvi delec v celici c; next[i] kaže na delec ki je naprej od delca i.
// Vstavljamo na začetek -> enosmerni povezan seznam za vsako celico.
static void celllist_build(CellList *cl, const Particle *particles,
                           unsigned int n)
{
    // 1. Inicializacija: vse celice prazne (-1 = konec seznama).
    for (int c = 0; c < cl->n_cells; c++)
        cl->head[c] = -1;

    // Za vsak delec
    for (unsigned int i = 0; i < n; i++)
    {
        // 2. Določi celico delca i iz njegovih koordinat.
        int cx = (int)(particles[i].x / cl->cell_size);
        int cy = (int)(particles[i].y / cl->cell_size);

        // 3. Zaščita pred robnim primerom, ko je x ali y == box_size.
        if (cx >= cl->nc)
            cx = cl->nc - 1;
        if (cy >= cl->nc)
            cy = cl->nc - 1;

        int c = cy * cl->nc + cx;

        // 4. Vstavi delec i na začetek seznama celice c
        cl->next[i] = cl->head[c]; // kateri je naprej od tega delca
        cl->head[c] = (int)i;      // kateri delec je v tej celici c

        // head[0] = 3
        // next[3] = 0
        // next[0] = -1
        // Seznam celice 0: 3 -> 0 -> -1
    }
}

// Force computation
// ena nit, en delec
// Vsaka vzporedna iteracija i obdela točno en delec i.
// Prebere sosede iz cell lista, silo akumulira v LOKALNI spremenljivki
// fix/fiy in jo ob koncu zapiše samo v particles[i].
// Ker nobena nit ne piše v delec druge niti, ni race conditionov in
// ni potrebe po atomics ali sinhronizaciji.
// Isti vzorec se direktno prenese na GPU (ena CUDA nit = en delec).

double compute_forces(Particle *particles, unsigned int n, double box_size)
{
    // 1. Konstante, ki so enake za vse pare
    const double v_shift = compute_v_shift();
    const double rc2 = R_CUT * R_CUT;       // kvadrat cutoff, da se izognemo sqrt()
    const double half_box = 0.5 * box_size; // za minimum-image konvencijo

    // 2. Zgradi cell list
    CellList cl = celllist_create(n, box_size);
    celllist_build(&cl, particles, n);

    double pe = 0.0;

    // 3. Vzporedna zanka: vsaka nit obdela en delec i neodvisno od ostalih.
#pragma omp parallel for reduction(+ : pe) schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        // Lokalni akumulator sile — vsaka nit piše samo sem, brez race conditiona.
        double fix = 0.0, fiy = 0.0;

        // 4. Določi celico delca i.
        int cx_i = (int)(particles[i].x / cl.cell_size);
        int cy_i = (int)(particles[i].y / cl.cell_size);
        if (cx_i >= cl.nc)
            cx_i = cl.nc - 1;
        if (cy_i >= cl.nc)
            cy_i = cl.nc - 1;

        // 5. Preglej 3×3 sosednje celice
        for (int dcy = -1; dcy <= 1; dcy++)
        {
            for (int dcx = -1; dcx <= 1; dcx++)
            {
                // Periodični ovoj indeksov celic: ((x % n) + n) % n deluje
                // pravilno tudi za negativne vrednosti (C % ne garantira tega).
                int cx_j = ((cx_i + dcx) % cl.nc + cl.nc) % cl.nc;
                int cy_j = ((cy_i + dcy) % cl.nc + cl.nc) % cl.nc;
                int cell_j = cy_j * cl.nc + cx_j;

                // 6. Preglej vse delce v celici prek povezanega seznama.
                for (int j = cl.head[cell_j]; j != -1; j = cl.next[j])
                {
                    if ((unsigned int)j == i)
                        continue; // preskoči self-interakcijo

                    // 7. Minimum-image konvencija: vzamemo najbližjo periodično kopijo j.
                    //    if/else je hitrejši kot nearbyint().
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

                    // 8. Preskoči pare zunaj cutoffa — brez sqrt().
                    double r2 = dx * dx + dy * dy;
                    if (r2 >= rc2 || r2 == 0.0)
                        continue;

                    // 9. Izračunaj LJ potence z množenjem namesto pow().
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;

                    // 10. Akumuliraj silo in PE za ta par.
                    //     0.5 pri PE: vsak par (i,j) obiščemo dvakrat (i→j in j→i).
                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij_r2 * dx;
                    fiy += fij_r2 * dy;
                    pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }

        // 11. Zapiši silo delca i — en sam vpis, nobena druga nit ne piše sem.
        particles[i].fx = fix;
        particles[i].fy = fiy;
    }

    celllist_free(&cl);
    return pe;
}

// Leapfrog je časovno reverzibilna shema (simplectic integrator), ki dobro
// ohranja energijo dolgoročno.
double leapfrog_step(Particle *particles, unsigned int n, double box_size)
{
    // Half-kick hitrosti + polni premik pozicije
    // to je združeno v eno zanko
    // Vsak delec neodvisen
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

    // Preračun sil iz novih pozicij — najdražji del
    double pe = compute_forces(particles, n, box_size);

    // Drugi half-kick z novimi silami
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    // vrne poračunano potencialno energijo sistema
    return pe;
}

// Top-level simulation runner

SimulationResult run_simulation(Particle *particles, unsigned int n,
                                unsigned int nsteps, double box_size,
                                int log_steps)
{
    SimulationResult out;

    // Začetne sile
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

    // Za vsak korak
    for (unsigned int step = 0; step < nsteps; step++)
    {
        // Premakne delce za en časovni korak dt in vrne PE iz novih pozicij.
        out.final_potential = leapfrog_step(particles, n, box_size);
        // KE izračunamo iz novih hitrosti
        out.final_kinetic = compute_ke(particles, n);
        // Skupaj s PE dobimo skupno energijo.
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
