#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=gpu_job_logs/convert_dataset%j.out

python convert2hf_dataset.py --data_dir "./data" --repeat_count 10 --output_name "zh_lora_dataset"