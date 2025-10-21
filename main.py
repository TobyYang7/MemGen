import os
import argparse
import torch
import random
import numpy as np
import logging
import time

from datetime import datetime

from larm.common.config import Config
from larm.common.logger import setup_logger
from larm.common.registry import registry
from larm.task import Task, BaseRunner

def set_seed(random_seed: int, use_gpu: bool):

    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

    print(f"set seed: {random_seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Language Reasoning and Memory")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def get_save_dir(config) -> str:
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("results", config.method, time)

def get_runner_class(config) -> BaseRunner:
    print(config.method)
    return registry.get_runner_class(config.method)

def main():
    
    # parse configs
    args = parse_args()
    config = Config(args)
    
    set_seed(config.run_cfg.seed, use_gpu=True)

    # set up save folder
    save_dir = get_save_dir(config)
    config.run_cfg.save_dir = save_dir

    # set up logger
    config.run_cfg.log_dir = os.path.join(save_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)

    config.pretty_print()
    
    task = Task(config)
    datasets_dict = task.build_dataset()
    env_and_gens_dict = task.build_env_and_generator()
    model = task.build_model()
    
    # build runner
    runner_cls = get_runner_class(config)
    runner = runner_cls(
        model=model, 
        processing_class=getattr(model, "processor", model.tokenizer), 
        configs=config,
        datasets_dict=datasets_dict, 
        env_and_gens_dict=env_and_gens_dict,
    )
    
    # train or evaluate
    if config.run_cfg.mode == "train":
        runner.train()
    if config.run_cfg.mode == "evaluate":
        runner.evaluate()

if __name__ == "__main__":
    main()