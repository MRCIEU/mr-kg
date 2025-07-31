#!/bin/bash

#SBATCH --job-name=embed-efo
#SBATCH --partition=mrcieu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=14G
#SBATCH --output=../data/output/slurm-%A_%a.out
#SBATCH --array=0

# NOTE: sbatch --account=<account-code> <script>
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

micromamba run -n mr-kg-processing \
  uv run scripts/embed-efo.py \
  --output-dir "${output_results_dir}" \
  >"${output_log_file}" 2>&1

date "+%Y-%m-%dT%H-%M"

