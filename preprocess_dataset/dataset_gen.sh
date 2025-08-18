#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=dataset_file_gen%j.out
python dataset_gen.py