#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gifenc.h"
#include "lennard-jones.h"

/*
 * CUDA implementacija Lennard-Jones simulacije z dvema glavnima optimizacijama:
 *
 * 1) Cell-list oziroma linked-cell optimizacija
 *    Simulacijski prostor razdelimo na kvadratne celice. Velikost celice je
 *    izbrana tako, da je vsaj R_CUT. Ker so Lennard-Jones interakcije zunaj
 *    radija R_CUT enake nič, mora delec preverjati samo delce v svoji celici
 *    in v sosednjih celicah. Tako se izognemo preverjanju vseh N*N parov.
 *
 * 2) Newtonov 3. zakon
 *    Interakcijo para (i, j) izračunamo samo enkrat. Sila, ki jo dodamo delcu i,
 *    je nasprotna sili, ki jo dodamo delcu j:
 *        F_ij = -F_ji
 *    Na GPE moramo zato pri zapisovanju sil uporabiti atomicAdd, saj lahko več
 *    CUDA niti hkrati posodablja silo istega delca.
 *
 * Opomba glede hitrosti:
 *    Ta različica je algoritmično bolj napredna, vendar so atomicAdd operacije
 *    lahko drage. Zato jo je smiselno primerjati tudi s preprostejšo različico,
 *    kjer en blok računa silo enega delca. Pri redkejših sistemih oziroma večjih
 *    N pa cell-list praviloma precej zmanjša število obravnavanih parov.
 */

#ifndef VECTOR_THREADS
#define VECTOR_THREADS 32
#endif

/* Uporablja se pri splošnih redukcijah, računanju kinetične energije in pack/unpack kernelih. */
#ifndef REDUCE_THREADS
#define REDUCE_THREADS 32
#endif

/*
 * Število niti na blok pri kernelu za izračun sil med pari celic.
 * En CUDA blok obdela en par celic. Niti znotraj bloka si razdelijo delo nad
 * pari delcev, nato pa z redukcijo seštejejo samo prispevek potencialne energije.
 */
#ifndef PAIR_THREADS
#define PAIR_THREADS 32
#endif

/*
 * Največje število delcev, ki jih hranimo v eni celici.
 * Pri pričakovani gostoti in velikosti celice okrog R_CUT bi morala biti ta
 * vrednost več kot dovolj. Če se pojavi opozorilo o overflowu celice, jo povečaj.
 */
#ifndef MAX_PARTICLES_PER_CELL
#define MAX_PARTICLES_PER_CELL 64
#endif

/*
 * Če je nastavljeno na 1, se po gradnji cell-list strukture iz GPE na CPE
 * prekopira en celoštevilski overflow flag. To je uporabno za razhroščevanje,
 * vendar pri vsakem izračunu sil doda majhen prenos, zato naj bo pri meritvah 0.
 */
#ifndef CHECK_CELL_OVERFLOW
#define CHECK_CELL_OVERFLOW 0
#endif

#define CELL_PAIR_TYPES 5

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
    img[(size_t)y * (size_t)w + (size_t)x] = index;
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

// -----------------------------------------------------------------------------
// Pomožne funkcije za CPE in združljivost z osnovno kodo
// -----------------------------------------------------------------------------

/* Enaka pomožna funkcija za naključna števila kot v osnovni kodi. Uporablja se samo pri inicializaciji. */
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

/* Javna CPE funkcija za kinetično energijo. Ohranimo jo zaradi združljivosti z originalnim vmesnikom. */
double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

/*
 * Inicializacija ostane na CPE, ker se izvede samo enkrat in ni ozko grlo.
 * Delce postavimo na mrežo, jim določimo naključne hitrosti, odstranimo povprečno
 * hitrost sistema in hitrosti skaliramo na zahtevano temperaturo.
 */
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

    for (unsigned int k = 0; k < n; ++k) {
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
    for (unsigned int k = 0; k < n; ++k) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; ++k) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);
        if (wx < 0.0) wx += box_size;
        if (wy < 0.0) wy += box_size;
        p->x = wx;
        p->y = wy;
    }
}

/* Premaknjen Lennard-Jones potencial pri radiju odreza. */
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

/*
 * Referenčni oziroma fallback izračun sil na CPE.
 * Tudi ta uporablja Newtonov 3. zakon: vsak par i<j izračunamo enkrat, nato pa
 * delcu j dodamo nasprotno silo. Funkcija se ne uporablja v run_simulation,
 * vendar jo ohranimo zaradi združljivosti z originalnim vmesnikom in za preverjanje.
 */
double compute_forces(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }

    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = compute_v_shift();
    double pe = 0.0;

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;

            if (dx > half_box) dx -= box_size;
            else if (dx < -half_box) dx += box_size;
            if (dy > half_box) dy -= box_size;
            else if (dy < -half_box) dy += box_size;

            double r2 = dx * dx + dy * dy;
            if (r2 < rcut2 && r2 > 0.0) {
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

/* CPE fallback za Leapfrog korak. Spodnja CUDA pot uporablja enakovredne GPE kernele. */
double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

// -----------------------------------------------------------------------------
// CUDA pomožne funkcije in kerneli
// -----------------------------------------------------------------------------

/* Periodično zavije indeks celice v interval [0, nc). */
static __device__ __forceinline__ int wrap_cell(int c, int nc) {
    if (c < 0) return c + nc;
    if (c >= nc) return c - nc;
    return c;
}

/*
 * Pretvori tabelo struktur Particle (AoS) v ločene tabele (SoA).
 * CPE/main uporablja strukture Particle, CUDA kerneli pa hitreje delajo z
 * ločenimi tabelami, ker sosednje niti berejo sosednje vrednosti x/y/v.
 */
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

/* Pretvori SoA tabele nazaj v strukture Particle za končni rezultat oziroma GIF. */
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

/*
 * Prva polovica Leapfrog koraka:
 *   v(t + dt/2) = v(t) + 0.5 * F(t) * dt
 *   x(t + dt)   = x(t) + v(t + dt/2) * dt
 * Ena CUDA nit posodobi en delec.
 */
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

    /* Periodični robni pogoji. Ker je DT majhen, je običajno dovolj en premik čez rob. */
    if (xi >= box_size) xi -= box_size;
    else if (xi < 0.0) xi += box_size;
    if (yi >= box_size) yi -= box_size;
    else if (yi < 0.0) yi += box_size;

    x[i] = xi;
    y[i] = yi;
    vx[i] = vxi;
    vy[i] = vyi;
}

/* Druga polovica Leapfrog koraka: v(t + dt) = v(t + dt/2) + 0.5 * F(t + dt) * dt. */
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

/* Pred izračunom novih sil ponastavi tabele sil. */
__global__ void clear_forces_kernel(double *__restrict__ fx,
                                    double *__restrict__ fy,
                                    unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fx[i] = 0.0;
    fy[i] = 0.0;
}

/* Ponastavi splošno tabelo double vrednosti; uporablja se za člene potencialne energije. */
__global__ void clear_double_kernel(double *__restrict__ values,
                                    unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    values[i] = 0.0;
}

/* Pred ponovno gradnjo cell-list strukture ponastavi število delcev v vsaki celici. */
__global__ void clear_cells_kernel(int *__restrict__ cell_counts,
                                   int *__restrict__ overflow,
                                   int total_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_cells) cell_counts[i] = 0;
    if (i == 0) *overflow = 0;
}

/*
 * Zgradi cell-list strukturo.
 * Vsak delec izračuna, v katero celico spada, in v to celico z atomicAdd doda
 * svoj indeks. Strukturo obnovimo v vsakem simulacijskem koraku, ker se delci premikajo.
 */
__global__ void build_cells_kernel(const double *__restrict__ x,
                                   const double *__restrict__ y,
                                   int *__restrict__ cell_counts,
                                   int *__restrict__ cell_particles,
                                   int *__restrict__ overflow,
                                   unsigned int n,
                                   int nc,
                                   double cell_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cx = (int)(x[i] / cell_size);
    int cy = (int)(y[i] / cell_size);

    /* Numerična varnost za koordinate, ki so zelo blizu box_size. */
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= nc) cx = nc - 1;
    if (cy >= nc) cy = nc - 1;

    int cell = cy * nc + cx;
    int slot = atomicAdd(&cell_counts[cell], 1);

    if (slot < MAX_PARTICLES_PER_CELL) {
        cell_particles[cell * MAX_PARTICLES_PER_CELL + slot] = (int)i;
    } else {
        atomicExch(overflow, 1);
    }
}

/*
 * Za podano celico in tip para vrne drugo celico v paru.
 * Uporabimo pet tipov parov, ker pri nc >= 3 z njimi vsak unikaten par sosednjih
 * celic pokrijemo natanko enkrat:
 *   0: ista celica      (0,  0)
 *   1: desna celica     (+1, 0)
 *   2: zgornja celica   (0, +1)
 *   3: zgoraj desno     (+1,+1)
 *   4: spodaj desno     (+1,-1)
 *
 * Teh pet smeri je dovolj, ker bi nasprotne smeri pomenile iste pare še enkrat.
 * Tukaj Newtonov 3. zakon odstrani podvojeno delo v primerjavi s preverjanjem
 * obeh smeri i->j in j->i.
 */
static __device__ __forceinline__ void decode_cell_pair(int cell,
                                                        int pair_type,
                                                        int nc,
                                                        int *cell_b) {
    int cy = cell / nc;
    int cx = cell - cy * nc;

    int ox = 0;
    int oy = 0;
    if (pair_type == 1) { ox = 1; oy = 0; }
    else if (pair_type == 2) { ox = 0; oy = 1; }
    else if (pair_type == 3) { ox = 1; oy = 1; }
    else if (pair_type == 4) { ox = 1; oy = -1; }

    int bx = wrap_cell(cx + ox, nc);
    int by = wrap_cell(cy + oy, nc);
    *cell_b = by * nc + bx;
}

/*
 * Kernel za izračun sil z uporabo cell-list strukture in Newtonovega 3. zakona.
 *
 * Razporeditev mreže:
 *   blockIdx.x določa en unikaten par celic.
 *   pair_index = cell * CELL_PAIR_TYPES + pair_type
 *
 * Delo znotraj enega bloka:
 *   - če je pair_type == 0, obdelamo unikatne pare znotraj iste celice: a < b
 *   - sicer obdelamo vse pare delcev med celico A in celico B
 *
 * Posodobitev sil:
 *   Za par (i, j) silo izračunamo samo enkrat. Nato z atomicAdd dodamo +F delcu i
 *   in -F delcu j. Atomske operacije so potrebne, ker lahko drug blok hkrati
 *   posodablja isti delec prek drugega sosednjega para celic.
 */
__global__ void compute_forces_newton_cell_kernel(const double *__restrict__ x,
                                                  const double *__restrict__ y,
                                                  double *__restrict__ fx,
                                                  double *__restrict__ fy,
                                                  double *__restrict__ pe_terms,
                                                  const int *__restrict__ cell_counts,
                                                  const int *__restrict__ cell_particles,
                                                  unsigned int n,
                                                  double box_size,
                                                  int nc) {
    unsigned int tid = threadIdx.x;
    unsigned int pair_index = blockIdx.x;
    int cell_a = (int)(pair_index / CELL_PAIR_TYPES);
    int pair_type = (int)(pair_index - (unsigned int)cell_a * CELL_PAIR_TYPES);

    int cell_b = cell_a;
    decode_cell_pair(cell_a, pair_type, nc, &cell_b);

    int count_a = cell_counts[cell_a];
    int count_b = cell_counts[cell_b];
    if (count_a > MAX_PARTICLES_PER_CELL) count_a = MAX_PARTICLES_PER_CELL;
    if (count_b > MAX_PARTICLES_PER_CELL) count_b = MAX_PARTICLES_PER_CELL;

    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = lj_v_shift();

    double local_pe = 0.0;

    if (pair_type == 0) {
        /* Ista celica: obdelamo samo pare a<b, da se izognemo podvajanju dela. */
        for (int a = 0; a < count_a; ++a) {
            int i = cell_particles[cell_a * MAX_PARTICLES_PER_CELL + a];
            for (int b = a + 1 + (int)tid; b < count_a; b += blockDim.x) {
                int j = cell_particles[cell_a * MAX_PARTICLES_PER_CELL + b];

                double dx = x[i] - x[j];
                double dy = y[i] - y[j];

                if (dx > half_box) dx -= box_size;
                else if (dx < -half_box) dx += box_size;
                if (dy > half_box) dy -= box_size;
                else if (dy < -half_box) dy += box_size;

                double r2 = dx * dx + dy * dy;
                if (r2 < rcut2 && r2 > 0.0) {
                    double inv_r2 = 1.0 / r2;
                    double sr2 = (SIGMA * SIGMA) * inv_r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                    double fij_x = f_over_r * dx;
                    double fij_y = f_over_r * dy;

                    atomicAdd(&fx[i], fij_x);
                    atomicAdd(&fy[i], fij_y);
                    atomicAdd(&fx[j], -fij_x);
                    atomicAdd(&fy[j], -fij_y);

                    local_pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
                }
            }
        }
    } else {
        /* Dve različni sosednji celici: vsi križni pari so unikatni. */
        int total_cross_pairs = count_a * count_b;
        for (int linear = (int)tid; linear < total_cross_pairs; linear += blockDim.x) {
            int a = linear / count_b;
            int b = linear - a * count_b;

            int i = cell_particles[cell_a * MAX_PARTICLES_PER_CELL + a];
            int j = cell_particles[cell_b * MAX_PARTICLES_PER_CELL + b];

            double dx = x[i] - x[j];
            double dy = y[i] - y[j];

            if (dx > half_box) dx -= box_size;
            else if (dx < -half_box) dx += box_size;
            if (dy > half_box) dy -= box_size;
            else if (dy < -half_box) dy += box_size;

            double r2 = dx * dx + dy * dy;
            if (r2 < rcut2 && r2 > 0.0) {
                double inv_r2 = 1.0 / r2;
                double sr2 = (SIGMA * SIGMA) * inv_r2;
                double sr6 = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;
                double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
                double fij_x = f_over_r * dx;
                double fij_y = f_over_r * dy;

                atomicAdd(&fx[i], fij_x);
                atomicAdd(&fy[i], fij_y);
                atomicAdd(&fx[j], -fij_x);
                atomicAdd(&fy[j], -fij_y);

                local_pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
            }
        }
    }

    /* Znotraj bloka reduciramo potencialno energijo; sile so bile že akumulirane z atomicAdd. */
    extern __shared__ double s_pe[];
    s_pe[tid] = local_pe;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_pe[tid] += s_pe[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pe_terms[pair_index] = s_pe[0];
    }
}

/*
 * Newtonov all-pairs fallback za zelo majhno število celic.
 * Pri nc < 3 se lahko zaradi periodičnega zavijanja sosednji pari celic podvojijo,
 * zato cell-list kernel izklopimo in uporabimo to natančno all-pairs različico.
 */
__global__ void compute_forces_newton_allpairs_kernel(const double *__restrict__ x,
                                                      const double *__restrict__ y,
                                                      double *__restrict__ fx,
                                                      double *__restrict__ fy,
                                                      double *__restrict__ pe_terms,
                                                      unsigned int n,
                                                      double box_size) {
    unsigned int i = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];
    const double half_box = 0.5 * box_size;
    const double rcut2 = R_CUT * R_CUT;
    const double v_shift = lj_v_shift();

    double local_pe = 0.0;

    for (unsigned int j = i + 1 + tid; j < n; j += blockDim.x) {
        double dx = xi - x[j];
        double dy = yi - y[j];

        if (dx > half_box) dx -= box_size;
        else if (dx < -half_box) dx += box_size;
        if (dy > half_box) dy -= box_size;
        else if (dy < -half_box) dy += box_size;

        double r2 = dx * dx + dy * dy;
        if (r2 < rcut2 && r2 > 0.0) {
            double inv_r2 = 1.0 / r2;
            double sr2 = (SIGMA * SIGMA) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * inv_r2;
            double fij_x = f_over_r * dx;
            double fij_y = f_over_r * dy;

            atomicAdd(&fx[i], fij_x);
            atomicAdd(&fy[i], fij_y);
            atomicAdd(&fx[j], -fij_x);
            atomicAdd(&fy[j], -fij_y);

            local_pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
        }
    }

    extern __shared__ double s_pe[];
    s_pe[tid] = local_pe;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_pe[tid] += s_pe[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pe_terms[i] = s_pe[0];
    }
}

/* Prispevek kinetične energije po delcih. Vsoto nato izračuna reduce_sum_kernel. */
__global__ void kinetic_terms_kernel(const double *__restrict__ vx,
                                     const double *__restrict__ vy,
                                     double *__restrict__ terms,
                                     unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    terms[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
}

/* Splošna redukcija po blokih: vsak blok reducira do 2*blockDim.x vhodnih vrednosti. */
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

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

/* Reducira tabelo, ki je že na GPE, in končni skalar vrne na CPE. */
static double reduce_device_sum(const double *d_values,
                                double *d_tmp1,
                                double *d_tmp2,
                                unsigned int n) {
    if (n == 0) return 0.0;

    const double *in = d_values;
    double *out = d_tmp1;
    unsigned int cur_n = n;

    while (cur_n > 1) {
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
                             unsigned int n) {
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
                            int total_cells) {
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
 * Funkcija izbere eno izmed dveh implementacij:
 *   - cell-list + Newtonov 3. zakon, kadar je nc >= 3
 *   - natančen all-pairs + Newton fallback, kadar je mreža celic premajhna
 */
static unsigned int compute_forces_gpu(const double *d_x,
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
                                       int use_cells) {
    unsigned int vector_blocks = (n + VECTOR_THREADS - 1) / VECTOR_THREADS;
    clear_forces_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());

    if (use_cells) {
        unsigned int total_cell_pairs = (unsigned int)total_cells * CELL_PAIR_TYPES;
        unsigned int term_blocks = (total_cell_pairs + VECTOR_THREADS - 1) / VECTOR_THREADS;
        clear_double_kernel<<<term_blocks, VECTOR_THREADS>>>(d_pe_terms, total_cell_pairs);
        CUDA_CHECK(cudaGetLastError());

        build_cells_gpu(d_x, d_y, d_cell_counts, d_cell_particles, d_overflow,
                        n, nc, cell_size, total_cells);

#if CHECK_CELL_OVERFLOW
        int h_overflow = 0;
        CUDA_CHECK(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_overflow) {
            fprintf(stderr, "Cell list overflow. Increase MAX_PARTICLES_PER_CELL.\n");
            exit(EXIT_FAILURE);
        }
#endif

        compute_forces_newton_cell_kernel<<<total_cell_pairs, PAIR_THREADS, PAIR_THREADS * sizeof(double)>>>(
            d_x, d_y, d_fx, d_fy, d_pe_terms, d_cell_counts, d_cell_particles,
            n, box_size, nc);
        CUDA_CHECK(cudaGetLastError());
        return total_cell_pairs;
    }

    /* Fallback: en blok na delec i, niti znotraj bloka obdelajo j>i. */
    clear_double_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_pe_terms, n);
    CUDA_CHECK(cudaGetLastError());

    compute_forces_newton_allpairs_kernel<<<n, PAIR_THREADS, PAIR_THREADS * sizeof(double)>>>(
        d_x, d_y, d_fx, d_fy, d_pe_terms, n, box_size);
    CUDA_CHECK(cudaGetLastError());
    return n;
}

// -----------------------------------------------------------------------------
// Glavna CUDA simulacija
// -----------------------------------------------------------------------------

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

    const size_t particle_bytes = (size_t)n * sizeof(Particle);
    const size_t double_bytes = (size_t)n * sizeof(double);
    const unsigned int vector_blocks = (n + VECTOR_THREADS - 1) / VECTOR_THREADS;

    /*
     * Velikost celice mora biti >= R_CUT, da so vsi možni interakcijski partnerji
     * v isti ali sosednjih celicah. Izberemo nc = floor(box/R_CUT), zato
     * cell_size = box/nc nikoli ni manjši od R_CUT.
     */
    int nc = (int)(box_size / R_CUT);
    if (nc < 1) nc = 1;
    double cell_size = box_size / (double)nc;
    int total_cells = nc * nc;

    /* Pri nc < 3 se periodični pari sosednjih celic lahko podvojijo, zato uporabimo natančen fallback. */
    int use_cells = (nc >= 3);
    unsigned int total_cell_pairs = use_cells ? (unsigned int)total_cells * CELL_PAIR_TYPES : n;

    /* d_pe_terms hrani člene potencialne energije, d_ke_terms pa člene kinetične energije. */
    unsigned int max_reduce_items = (total_cell_pairs > n) ? total_cell_pairs : n;
    const size_t pe_bytes = (size_t)max_reduce_items * sizeof(double);
    const unsigned int reduce_capacity = (max_reduce_items + (REDUCE_THREADS * 2 - 1)) / (REDUCE_THREADS * 2);
    const size_t reduce_bytes = (size_t)((reduce_capacity > 1) ? reduce_capacity : 1) * sizeof(double);

    Particle *d_particles = NULL;
    double *d_x = NULL, *d_y = NULL, *d_vx = NULL, *d_vy = NULL, *d_fx = NULL, *d_fy = NULL;
    double *d_pe_terms = NULL, *d_ke_terms = NULL, *d_tmp1 = NULL, *d_tmp2 = NULL;
    int *d_cell_counts = NULL, *d_cell_particles = NULL, *d_overflow = NULL;

    /* Vse trajne GPE tabele alociramo enkrat. Med celotno simulacijo ostanejo na GPE. */
    CUDA_CHECK(cudaMalloc((void **)&d_particles, particle_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vx, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_vy, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fx, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_fy, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_pe_terms, pe_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ke_terms, double_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp1, reduce_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_tmp2, reduce_bytes));

    if (use_cells) {
        CUDA_CHECK(cudaMalloc((void **)&d_cell_counts, (size_t)total_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_cell_particles,
                              (size_t)total_cells * MAX_PARTICLES_PER_CELL * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_overflow, sizeof(int)));
    }

    /*
     * Začetni prenos CPE -> GPE. Ker main.c meri čas okoli run_simulation(),
     * je tudi ta prenos vključen v čas meritev.
     */
    CUDA_CHECK(cudaMemcpy(d_particles, particles, particle_bytes, cudaMemcpyHostToDevice));

    /* Pretvorimo Particle strukture v SoA tabele, ki jih uporabljajo CUDA kerneli. */
    pack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());

    /* Izračunamo začetne sile in začetno potencialno energijo. Sile potrebujemo za prvi Leapfrog korak. */
    unsigned int pe_count = compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_terms,
                                               d_cell_counts, d_cell_particles, d_overflow,
                                               n, box_size, nc, cell_size, total_cells, use_cells);
    out.start_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, pe_count);
    out.start_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
    out.start_total = out.start_kinetic + out.start_potential;

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

#if GENERATE_GIF
    ge_GIF *gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; ++step) {
        /* 1) Prva polovica Leapfrog koraka: posodobimo hitrosti in pozicije. */
        integrate_first_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_x, d_y, d_vx, d_vy,
                                                                  d_fx, d_fy, n, box_size);
        CUDA_CHECK(cudaGetLastError());

        /* 2) Ponovno zgradimo cell-list in izračunamo nove sile pri posodobljenih pozicijah. */
        pe_count = compute_forces_gpu(d_x, d_y, d_fx, d_fy, d_pe_terms,
                                      d_cell_counts, d_cell_particles, d_overflow,
                                      n, box_size, nc, cell_size, total_cells, use_cells);

        /* 3) Druga polovica Leapfrog koraka: dokončamo posodobitev hitrosti z novimi silami. */
        integrate_second_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_vx, d_vy, d_fx, d_fy, n);
        CUDA_CHECK(cudaGetLastError());

        /* Opcijsko beleženje energije. Pri benchmark meritvah naj bo log_steps=0. */
        if (log_steps) {
            out.final_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, pe_count);
            out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
            out.final_total = out.final_kinetic + out.final_potential;
            printf("step=%6u KE=%12.6f PE=%12.6f E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            unpack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(particles, d_particles, particle_bytes, cudaMemcpyDeviceToHost));
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

    /* Če energije nismo beležili v vsakem koraku, jo na koncu izračunamo samo enkrat. */
    if (!log_steps && nsteps > 0) {
        out.final_potential = reduce_device_sum(d_pe_terms, d_tmp1, d_tmp2, pe_count);
        out.final_kinetic = compute_ke_gpu(d_vx, d_vy, d_ke_terms, d_tmp1, d_tmp2, n);
        out.final_total = out.final_kinetic + out.final_potential;
    }

    /*
     * Končni prenos delcev GPE -> CPE. Tudi ta prenos je znotraj run_simulation(),
     * zato je vključen v izmerjeni čas GPE izvajanja.
     */
    unpack_particles_kernel<<<vector_blocks, VECTOR_THREADS>>>(d_particles, d_x, d_y, d_vx, d_vy, d_fx, d_fy, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(particles, d_particles, particle_bytes, cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaFree(d_pe_terms));
    CUDA_CHECK(cudaFree(d_ke_terms));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));
    if (d_cell_counts) CUDA_CHECK(cudaFree(d_cell_counts));
    if (d_cell_particles) CUDA_CHECK(cudaFree(d_cell_particles));
    if (d_overflow) CUDA_CHECK(cudaFree(d_overflow));

    return out;
}
