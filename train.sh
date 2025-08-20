#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=gpu_job_logs/train%j.out

#general
NUM_NODES=1
SHIFT=3.0

#training hyperparameters
LEARNING_RATE=1e-4
NUM_WORKERS=8
EPOCHS=1
MAX_STEPS=2000000
EVERY_N_TRAIN_STEPS=200

#experiment settings
DATASET_PATH="./zh_lora_dataset"
EXP_NAME="transition_generation"

#training precision and gradient settings
PRECISION="32"
ACCUMULATE_GRAD_BATCHES=1
GRADIENT_CLIP_VAL=0.5
GRADIENT_CLIP_ALGORITHM="norm"

#checkpoint and logging setttings
DEVICES=1
LOGGER_DIR="./exps/logs/"
CKPT_PATH="none"
CHECKPOINT_DIR="/vol/bitbucket/al4624/ace_step_model_output/"

#Validation and reloading settings
RELOAD_DATALOADERS_EVERY_N_EPOCHS=1
EVERY_PLOT_STEP=4000
VAL_CHECK_INTERVAL=30
LORA_CONFIG_PATH="config/zh_rap_lora_config.json"

CMD="python trainer.py --num_nodes $NUM_NODES \
                       --shift $SHIFT \
                       --learning_rate $LEARNING_RATE \
                       --num_workers $NUM_WORKERS \
                       --epochs $EPOCHS \
                       --max_steps $MAX_STEPS \
                       --every_n_train_steps $EVERY_N_TRAIN_STEPS \
                       --dataset_path $DATASET_PATH \
                       --exp_name $EXP_NAME \
                       --precision $PRECISION \
                       --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
                       --gradient_clip_val $GRADIENT_CLIP_VAL \
                       --gradient_clip_algorithm $GRADIENT_CLIP_ALGORITHM \
                       --devices $DEVICES \
                       --logger_dir $LOGGER_DIR \
                       --val_check_interval $VAL_CHECK_INTERVAL \
                       --checkpoint_dir $CHECKPOINT_DIR \
                       --reload_dataloaders_every_n_epochs $RELOAD_DATALOADERS_EVERY_N_EPOCHS \
                       --every_plot_step $EVERY_PLOT_STEP \
                       --lora_config_path $LORA_CONFIG_PATH"


if [ "$CKPT_PATH" != "none" ]; then
    CMD="$CMD --ckpt_path $CKPT_PATH"
fi

echo $CMD
eval $CMD
