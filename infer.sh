#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=gpu_job_logs/run_infer%j.out


python infer.py --output_path /homes/al4624/Documents/YuE_finetune/inference_audio_prompts/generated.wav \
                --start_audio_path /homes/al4624/Documents/YuE_finetune/inference_audio_prompts/start.mp3 \
                --end_audio_path /homes/al4624/Documents/YuE_finetune/inference_audio_prompts/end.mp3 \
                --concat_audio_path /homes/al4624/Documents/YuE_finetune/inference_audio_prompts/full.mp3 \
                --repaint_variance 0.8 \
                --gen_duration 10.0 \