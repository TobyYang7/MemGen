#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./test_output/debug_log.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_FILE_ONLY=0

# Wandb configuration (optional)
# export WANDB_MODE="offline"  # offline mode: saves locally, sync later with "wandb sync"
export WANDB_PROJECT="memgen-training"
export WANDB_NAME="mm_math_sft"

# DATASET_NAME="math_vision"
# DATASET_NAME="mmvp"
DATASET_NAME="mm_math"

# train
uv run python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes 1 \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    run.train_weaver_method sft \
    model.reasoner_model_name Qwen/Qwen2.5-VL-7B-Instruct \
    model.weaver.weaver_model_name Qwen/Qwen2.5-1.5B-Instruct \
    model.trigger.trigger_model_name null \
    model.weaver.prompt_latents_len 8 \
    model.weaver.inference_latents_len 8 \
    model.max_prompt_aug_num 1 \
    model.max_inference_aug_num 5 \
    run.weaver.sft.max_epochs 2 \
    model.load_model_path null \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.generation.do_sample True \
    run.generation.temperature 1.0 \
    datasets.${DATASET_NAME}.mode sft \
    run.weaver.sft.batch_size 1 \
    model.use_entropy_filter True \
    model.entropy_threshold 0.7 \
    run.output_dir /root/toby/MemGen/test_output/${DATASET_NAME} \