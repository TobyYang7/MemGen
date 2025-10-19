uv venv --python 3.10
uv pip install torch
uv pip install trl
uv pip install transformers datasets vllm accelerate codetiming datasets dill hydra-core numpy pandas pybind11 ray wandb deepspeed setuptools peft tensorboard pyserini uvicorn fastapi
FLASH_ATTN_CUDA_ARCHS="90" uv pip install flash-attn --no-build-isolation
