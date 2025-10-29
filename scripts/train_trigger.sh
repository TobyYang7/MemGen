#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 
TRIGGER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Dataset configs
DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_MODE="grpo"   # options: sft or grpo

# MemGen configs
TRAIN_METHOD="grpo"   # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Trained weaver model path: 
# - Must point to a checkpoint file ending with .safetensors (e.g. <output_dir>/model.safetensors)
# - Required when training the trigger (a pre-trained weaver model must exist)
LOAD_WEAVER_PATH="<output_dir>/model.safetensors"

# train
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.reasoner_model_name ${REASONER_MODEL} \
    model.weaver.weaver_model_name ${WEAVER_MODEL} \
    model.trigger.trigger_model_name ${TRIGGER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
    datasets.${DATASET_NAME}.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.generation.do_sample True \
    run.generation.temperature 1.0 \
    run.generation.max_response_length 512 \





