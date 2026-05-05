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
// GIF rendering (unchanged from reference)
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

// Kinetična energija: Ek = sum_i 0.5 * |v_i|^2
// Obremenitev je enakomerna (vsak delec enako dela), zato static urnik.
// reduction(+:ke) zagotavlja, da vsaka nit sešteva v svojo lokalno kopijo,
// ki se ob koncu seštejejo skupaj brez race conditiona.
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
// Particle initialisation (unchanged logic)
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

// ---------------------------------------------------------------------------
// Periodic boundary conditions
// ---------------------------------------------------------------------------

// Periodični robni pogoji: delci, ki zapustijo škatlo, se pojavijo na
// nasprotni strani. fmod vrne vrednost v [-box_size, box_size], zato
// negativne vrednosti popravimo z dodatkom box_size.
// Vsak delec je neodvisen → static urnik (enaka obremenitev, brez overhead-a).
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
// Shifted LJ potential: V_shifted(r) = V(r) - V(r_cut)
// ---------------------------------------------------------------------------

// LJ potencial pri r = R_CUT ni točno 0 (ampak ~0.016*epsilon), kar bi
// povzročilo skok pri prehodu čez mejo cutoff-a. To krši ohranjanje energije.
// Shift odšteje V(R_CUT) od vsakega para, tako da potencial gladko pade na 0.
// Izračunamo enkrat pred zanko, da ne ponavljamo dragih množenj.
// sr6 = (sigma/R_CUT)^6 z množenji namesto pow() – hitrejše.
double compute_v_shift(void)
{
    double sr_cut = SIGMA / R_CUT;
    double sr6 = sr_cut * sr_cut * sr_cut * sr_cut * sr_cut * sr_cut;
    return 4.0 * EPSILON * (sr6 * sr6 - sr6);
}

// ---------------------------------------------------------------------------
// Cell list
//
// Referenčna koda ima O(N²) zanko – vsak delec primerja z vsakim.
// Cell list razdeli škatlo na mrežo celic s stranico >= R_CUT.
// Ker delci zunaj R_CUT ne vplivajo drug na drugega, vsak delec i
// pregleda samo 3×3 = 9 sosednjih celic namesto vseh N delcev.
// Skupna kompleksnost pade z O(N²) na O(N) – potrjeno z meritvami:
//   N=1000 → 0.528 s,  N=2000 → 0.737 s,  N=4000 → 1.116 s,  N=8000 → 1.634 s
// (podvojitev N ≈ 1.5× časa namesto 4× pri O(N²))
// Cell list se obnovi vsak korak, ker se delci premikajo.
// ---------------------------------------------------------------------------

typedef struct
{
    int *head; // head[c] = indeks prvega delca v celici c, -1 če prazna
    int *next; // next[i] = naslednji delec v isti celici, -1 če zadnji
    int nc;    // število celic na stran
    double cell_size;
    int n_cells;
} CellList;

static CellList celllist_create(unsigned int n, double box_size)
{
    CellList cl;
    // nc = floor(box_size / R_CUT) → cell_size >= R_CUT vedno drži.
    // To je ključno: celica mora biti vsaj R_CUT velika, sicer bi partner
    // lahko ležal 2+ celic stran in bi ga 3×3 pregled zgrešil.
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

// Gradi povezan seznam delcev po celicah z vstavljanjem na začetek (O(N)).
// head[c] kaže na zadnji vstavljeni delec v celici c; next[i] kaže na
// prejšnjega. Rezultat je enosmerni seznam za vsako celico brez dodatnega
// pomnilnika za fiksno dolge sezname.
static void celllist_build(CellList *cl, const Particle *particles,
                           unsigned int n)
{
    for (int c = 0; c < cl->n_cells; c++)
        cl->head[c] = -1;
    for (unsigned int i = 0; i < n; i++)
    {
        int cx = (int)(particles[i].x / cl->cell_size);
        int cy = (int)(particles[i].y / cl->cell_size);
        // Zaščita pred floating-point robnim primerom, ko je x/y == box_size
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
// Force computation
//
// Osnovna ideja – "ena nit, en delec":
//
//   Vsaka vzporedna iteracija i obdela točno en delec i.
//   Prebere sosede iz cell lista, silo akumulira v LOKALNI spremenljivki
//   fix/fiy in jo ob koncu zapiše samo v particles[i].
//
//   Ker nobena nit ne piše v delec druge niti, ni race conditionov in
//   ni potrebe po atomics ali sinhronizaciji.
//   Isti vzorec se direktno prenese na GPU (ena CUDA nit = en delec).
//
//   Kompromis: par (i,j) obiščemo dvakrat (enkrat za i, enkrat za j),
//   zato pri PE upoštevamo faktor 0.5. Newton 3. zakon bi prepolovil
//   delo, a bi zahteval atomics ali race-condition-free akumulacijo,
//   kar bi zakompliciralo vzporeditev (posebej na GPU).
// ---------------------------------------------------------------------------

double compute_forces(Particle *particles, unsigned int n, double box_size)
{
    const double v_shift = compute_v_shift();
    // Primerjamo r² z rc² namesto r z R_CUT – prihranimo sqrt() za vsak par.
    const double rc2 = R_CUT * R_CUT;
    const double half_box = 0.5 * box_size;

    CellList cl = celllist_create(n, box_size);
    celllist_build(&cl, particles, n);

    double pe = 0.0;

    // static urnik: meritve so pokazale, da pri naših velikostih sistema je to bolje
#pragma omp parallel for reduction(+ : pe) schedule(static)
    for (unsigned int i = 0; i < n; ++i)
    {
        // Lokalna akumulatorja sile: vsaka nit piše le sem, brez deljenja
        // pomnilnika z drugimi nitmi → brez false sharinga, brez race conditiona.
        double fix = 0.0, fiy = 0.0;

        int cx_i = (int)(particles[i].x / cl.cell_size);
        int cy_i = (int)(particles[i].y / cl.cell_size);
        if (cx_i >= cl.nc)
            cx_i = cl.nc - 1;
        if (cy_i >= cl.nc)
            cy_i = cl.nc - 1;

        // Pregledamo 3×3 celice v okolici (s periodičnim ovojem prek modulo).
        // Ker je cell_size >= R_CUT, so garantirani vsi možni partnerji znotraj
        // R_CUT v teh 9 celicah – nobeden ne leži dlje.
        for (int dcy = -1; dcy <= 1; dcy++)
        {
            for (int dcx = -1; dcx <= 1; dcx++)
            {
                // Periodični ovoj indeksov celic: ((x % n) + n) % n deluje
                // pravilno tudi za negativne vrednosti (C % ne garantira tega).
                int cx_j = ((cx_i + dcx) % cl.nc + cl.nc) % cl.nc;
                int cy_j = ((cy_i + dcy) % cl.nc + cl.nc) % cl.nc;
                int cell_j = cy_j * cl.nc + cx_j;

                for (int j = cl.head[cell_j]; j != -1; j = cl.next[j])
                {
                    if ((unsigned int)j == i)
                        continue; // preskoči self-interakcijo

                    // Minimum-image konvencija: izberemo najbližjo periodično
                    // sliko delca j glede na i. if/else je hitrejši kot
                    // nearbyint(), ki ga uporablja referenčna koda.
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
                    // Preskoči pare zunaj cutoff-a brez sqrt – preverimo r² ≥ rc².
                    if (r2 >= rc2 || r2 == 0.0)
                        continue;

                    // (σ/r)^6 in (σ/r)^12 z množenji namesto pow().
                    // pow() kliče exp/log internalno in je ~10× počasnejši
                    // od navadnih množenj za cele potence.
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;

                    // fij / r² = F(r) / r: sila v smeri dx, dy brez sqrt.
                    // Referenčna koda: fij/r * (dx/r) = fij*dx/r² – enako,
                    // a tam je bil r = sqrt(r²) izračunan eksplicitno.
                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij_r2 * dx;
                    fiy += fij_r2 * dy;

                    // 0.5: ker vsak par (i,j) obiščemo dvakrat (i→j in j→i).
                    pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }

        // En sam vpis na delec i ob koncu – nobena druga nit ne piše sem.
        particles[i].fx = fix;
        particles[i].fy = fiy;
    }

    celllist_free(&cl);
    return pe;
}

// Leapfrog je časovno reverzibilna shema (simplectic integrator), ki dobro
// ohranja energijo dolgoročno. En korak sestoji iz:
//   1. half-kick: v(t + dt/2) = v(t) + 0.5*a(t)*dt
//   2. drift:     r(t + dt)   = r(t) + v(t + dt/2)*dt
//   3. sile:      a(t + dt)   iz novih pozicij
//   4. half-kick: v(t + dt)   = v(t + dt/2) + 0.5*a(t+dt)*dt
//
// Koraka 1+2 sta združena v eno zanko (v referenčni kodi sta ločeni).

double leapfrog_step(Particle *particles, unsigned int n, double box_size)
{
    // Koraka 1+2 združena: half-kick hitrosti + polni premik pozicije.
    // Obremenitev je enakomerna → static urnik.
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

    // Preračun sil iz novih pozicij – najdražji del koraka.
    double pe = compute_forces(particles, n, box_size);

    // Korak 4: drugi half-kick z novimi silami.
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
