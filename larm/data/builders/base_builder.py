from datasets import DatasetDict
from typing import Dict, Literal
from omegaconf import OmegaConf
from abc import ABC, abstractmethod

from larm.common import utils
from larm.data.envs.base_env import BaseEnv

class BaseDatasetBuilder(ABC):

    def __init__(self, cfg: Dict = None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            config = load_dataset_config(cfg)
        else:
            # when called from runner.build_dataset()
            config = cfg
        
        self.mode = config.get("mode", "sft")
        self.full_config = config  # 保存完整配置以访问父级参数
        self.config = config.get(self.mode)

    def build_datasets(self) -> DatasetDict:
        method_builder_map = {
            "sft": self._build_sft_datasets,
            "grpo": self._build_rl_datasets,
        }

        if self.mode not in method_builder_map:
            raise ValueError("Unsupported datasets mode")
        
        return method_builder_map[self.mode]()

    @abstractmethod
    def _build_sft_datasets(self) -> DatasetDict:
        raise NotImplementedError("Should be implemented by subclasses")
    
    @abstractmethod
    def _build_rl_datasets(self) -> DatasetDict:
        raise NotImplementedError("Should be implemented by subclasses")
    
    @abstractmethod
    def get_env_cls(self) -> BaseEnv:
        raise NotImplementedError("Should be implemented by subclasses")

    @abstractmethod
    def get_generation_manager_cls(self) -> BaseEnv:
        raise NotImplementedError("Should be implemented by subclasses")
    
    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])
    

def load_dataset_config(cfg_path: str) -> Dict:
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg
