#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./test_output/eval_json_debug_log.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_FILE_ONLY=1

# ===== Configuration =====

# Base model evaluation (optional)
# Set to "true" to evaluate only the base reasoner model without weaver/trigger
# Set to "false" or leave empty to evaluate the full trained model
BASE_MODEL="false"

# JSON file to evaluate (REQUIRED)
# This should be a JSON file with format: [{"prompt": "...", "solution": "...", "image_path": "..."}]
JSON_PATH=/root/toby/MemGen/data/math_vision/test.json

# Output directory
OUTPUT_DIR=/root/toby/MemGen/eval/math_vision

# Output filename (optional)
# If not specified, will auto-generate based on model type
# Examples: "answer.json", "my_results.json"
OUTPUT_FILENAME=""

# Trained model path (REQUIRED)
# Must point to a checkpoint file ending with .safetensors
MODEL_PATH=/root/toby/MemGen/test_output/math_vision/weaver/model.safetensors

# Model names
# REASONER_MODEL="UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B"
REASONER_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL=""  # Leave empty for no trigger model

# Augmentation configuration
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Generation configuration
BATCH_SIZE=32
DO_SAMPLE=""  # Add "--do_sample" to enable sampling, leave empty for greedy
TEMPERATURE=1.0
MAX_RESPONSE_LENGTH=1024

# ===== Run Evaluation =====

echo "Starting evaluation from JSON file..."
echo "JSON Path: ${JSON_PATH}"
echo "Base Model: ${BASE_MODEL}"
if [ "${BASE_MODEL}" = "true" ]; then
    echo "Mode: Base Model Evaluation (no weaver/trigger)"
    echo "Reasoner: ${REASONER_MODEL}"
else
    echo "Mode: Full Model Evaluation (with weaver/trigger)"
    echo "Model Path: ${MODEL_PATH}"
fi
echo "Output Dir: ${OUTPUT_DIR}"

# Build command based on BASE_MODEL flag
if [ "${BASE_MODEL}" = "true" ]; then
    # Base model evaluation (no weaver/trigger needed)
    uv run python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        eval_from_json.py \
        --json_path ${JSON_PATH} \
        --reasoner_model ${REASONER_MODEL} \
        --weaver_model ${WEAVER_MODEL} \
        --model_path ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        ${DO_SAMPLE} \
        --temperature ${TEMPERATURE} \
        --max_response_length ${MAX_RESPONSE_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        ${OUTPUT_FILENAME:+--output_filename ${OUTPUT_FILENAME}} \
        --base_model
else
    # Full model evaluation (with weaver/trigger)
    uv run python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        eval_from_json.py \
        --json_path ${JSON_PATH} \
        --model_path ${MODEL_PATH} \
        --reasoner_model ${REASONER_MODEL} \
        --weaver_model ${WEAVER_MODEL} \
        ${TRIGGER_MODEL:+--trigger_model ${TRIGGER_MODEL}} \
        --max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        --max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        --prompt_latents_len ${PROMPT_LATENTS_LEN} \
        --inference_latents_len ${INFERENCE_LATENTS_LEN} \
        --batch_size ${BATCH_SIZE} \
        ${DO_SAMPLE} \
        --temperature ${TEMPERATURE} \
        --max_response_length ${MAX_RESPONSE_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        ${OUTPUT_FILENAME:+--output_filename ${OUTPUT_FILENAME}}
fi

if [ -n "${OUTPUT_FILENAME}" ]; then
    echo "Evaluation completed! Check results in ${OUTPUT_DIR}/${OUTPUT_FILENAME}"
else
    echo "Evaluation completed! Check results in ${OUTPUT_DIR}/ (filename auto-generated)"
fi

