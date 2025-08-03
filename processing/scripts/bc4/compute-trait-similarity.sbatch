#!/bin/bash

#SBATCH --job-name=compute-trait-similarity
#SBATCH --partition=mrcieu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=../data/output/slurm-%A_%a.out
#SBATCH --array=0-19

# NOTE: sbatch --account=<account-code> <script>
# This job uses read-only database access, so multiple tasks can run concurrently
# without database conflicts. Each task processes a different chunk of PMID-model combinations.
# Each task uses 8 CPU cores for multiprocessing within the chunk processing.
date "+%Y-%m-%dT%H-%M"

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

prefix="bc4"
output_dir="${SLURM_SUBMIT_DIR}"/../data/output/"${prefix}"-"${SLURM_ARRAY_JOB_ID}"
mkdir -p "${output_dir}"/{results,logs}
output_results_dir="${output_dir}"/results

output_log_file="${output_dir}"/logs/slurm-"${SLURM_ARRAY_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}".out
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  submit_script_log_file="${output_dir}"/logs/script-"${SLURM_ARRAY_JOB_ID}".out
  scontrol write batch_script "${SLURM_ARRAY_JOB_ID}" ${submit_script_log_file}
fi


echo "Using database: $database_path"

micromamba run -n mr-kg-processing \
  uv run scripts/compute-trait-similarity.py \
  --array-length 20 \
  --array-id "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${output_results_dir}" \
  --database-path "${SLURM_SUBMIT_DIR}"/../data/db/vector_store.db \
  --top-k 10 \
  --workers 8 \
  >"${output_log_file}" 2>&1

date "+%Y-%m-%dT%H-%M"
