#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --job-name=esm2_job

input_fasta=$(realpath $1)
output_dir=$(realpath $2)
esm_model_id=$3

echo $output_dir

esm_models=/shared/archive/cas/common-data/esm-models

docker run --user $(id -u):$(id -g) \
           -v ${input_fasta}:/workspace/input-fasta.fa \
           -v ${output_dir}:/workspace/output_dir \
           -v ${esm_models}:/esm-models \
           -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
           --rm --gpus all \
           esm2 --model-dir /esm-models/${esm_model_id} --out-dir /workspace/output_dir --save-per-residue --pooling none /workspace/input-fasta.fa
