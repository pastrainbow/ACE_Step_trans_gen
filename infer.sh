#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=run_infer%j.out

# torch_compile=true # @param {type: "boolean"}
# cpu_offload=False # @param {type: "boolean"}
# overlapped_decode = True # @param {type: "boolean"}
#bf16 = True # @param {type: "boolean"}

# acestep --checkpoint_path /unzip/checkpoints/ \
#         --port 7865 \
#         --device_id 0 \
#         --share true \
#         --torch_compile true \
#         --cpu_offload false \
#         --overlapped_decode true

# stdbuf -oL -eL acestep --port 7865 \
#                        --share true \

python infer.py