#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./test_output/eval_debug_log.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_FILE_ONLY=1

DATASET_NAME="mmvp"

# Trained model path: 
# - Must point to a checkpoint file ending with .safetensors (e.g. <output_dir>/weaver/model.safetensors)
# - Required when evaluating the model
LOAD_MODEL_PATH="/root/toby/MemGen/test_output/${DATASET_NAME}/weaver/model.safetensors"

# Evaluation JSON path (optional):
# - If specified, data will be loaded from this JSON file instead of the default data loader
# - The JSON file should have the same format as the preprocessed data
# - Example: "/root/toby/MemGen/data/mmvp/test.json"
EVAL_JSON_PATH="/root/toby/MemGen/data/mmvp/test.json"

BASE_MODEL=True

# evaluate
uv run python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.reasoner_model_name UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B \
    model.weaver.weaver_model_name Qwen/Qwen2.5-1.5B-Instruct \
    model.trigger.trigger_model_name null \
    model.weaver.prompt_latents_len 8 \
    model.weaver.inference_latents_len 8 \
    model.max_prompt_aug_num 1 \
    model.max_inference_aug_num 3 \
    model.load_model_path ${LOAD_MODEL_PATH} \
    run.mode evaluate \
    run.generation.eval_batch_size 8 \
    run.generation.do_sample False \
    run.generation.temperature 1.0 \
    run.generation.max_response_length 256 \
    run.output_dir /root/toby/MemGen/test_output/${DATASET_NAME}_eval \
    datasets.${DATASET_NAME}.mode grpo \
    datasets.${DATASET_NAME}.grpo.eval_json_path ${EVAL_JSON_PATH}

