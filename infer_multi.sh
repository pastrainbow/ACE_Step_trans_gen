#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=gpu_job_logs/run_infer_multi%j.out


python infer_multi.py --output_dir /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/generated \
                --full_audio_dir /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/full_audio \
                --audio_prompt_dir /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/split_audio_prompts \
                --concat_audio_dir /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/concat_noised \
                --repaint_variance 0.8 \
                --gen_duration 10.0 \