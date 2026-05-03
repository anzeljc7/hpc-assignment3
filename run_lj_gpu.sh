#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=slurm_temp_%j.log
#SBATCH --time=00:15:00

module load CUDA

make clean
make

# ==========================================
# NASTAVITVE MERITEV
# ==========================================
PROGRAM_NAME="gpu-timotej2-8"
RUNS=5
N=1000
NSTEPS=5000

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RESULTS_DIR="results/${PROGRAM_NAME}"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Začenjam GPU meritve..."
echo "Testiram mrežo: ${N} delcev, ${NSTEPS} korakov"
echo "=========================================="

# Spremenljivka za shranjevanje vseh časov
ALL_TIMES=""

for ((i=1; i<=RUNS; i++)); do
    echo "--- Zagon $i/$RUNS ---"
    
    # Zaženemo program in zajamemo celoten izpis
    program_output=$(srun ./lj.out $N $NSTEPS 2>&1)
    
    # Izpišemo celoten output programa v log
    echo "$program_output"
    
    # Izluščimo samo čas
    time_s=$(echo "$program_output" | awk '/^Simulation time/ {print $5}')

    if [ -z "$time_s" ]; then
        echo "  Napaka: časa nisem našel v izpisu za N=$N."
        exit 1
    fi

    printf "\n--> Izmerjen čas za zagon %d: %9s s\n\n" "$i" "$time_s"
    ALL_TIMES="$ALL_TIMES $time_s"
done

# Izračun povprečja
avg=$(echo "$ALL_TIMES" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; if(NF>0) printf "%.6f", sum/NF}')

echo "=========================================="
printf "  POVPREČJE ZA N=%-4s = %9s s\n" "$N" "$avg"
echo "=========================================="

# Preimenujemo in premaknemo začasno log datoteko v results mapo
FINAL_SLURM_LOG="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}.log"
mv "slurm_temp_${SLURM_JOB_ID}.log" "$FINAL_SLURM_LOG"

echo "Log datoteka je shranjena v: $FINAL_SLURM_LOG"