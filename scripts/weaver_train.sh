#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./test_output/debug_log_mm_math.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# train
uv run python -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port=29507 \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/mm_math.yaml \
    --options \
    model.reasoner_model_name Qwen/Qwen2.5-VL-3B-Instruct \
    model.weaver.weaver_model_name Qwen/Qwen2.5-0.5B-Instruct \
    model.trigger.trigger_model_name null \
    model.weaver.prompt_latents_len 4 \
    model.weaver.inference_latents_len 4 \
    model.max_prompt_aug_num 1 \
    model.max_inference_aug_num 1 \
    model.load_model_path null \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method grpo \
    run.generation.do_sample True \
    run.generation.temperature 1.0 \
    run.generation.max_response_length 512 \
    run.output_dir /root/toby/MemGen/test_output/mm_math \
    datasets.mm_math.mode grpo