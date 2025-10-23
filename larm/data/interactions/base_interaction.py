
from typing import Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from larm.data.utils.tensor_utils import TensorHelper, TensorConfig


@dataclass
class InteractionConfig:
    max_turns: int
    max_start_length: int   
    max_prompt_length: int   
    max_response_length: int
    max_obs_length: int
    do_sample: bool
    temperature: float  

@dataclass
class InteractionDataProto:
    batch: Dict = field(default_factory=dict)
    no_tensor_batch: Dict = field(default_factory=dict)

class InteractionManager(ABC):
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        tokenizer = tokenizer.tokenizer # fix: processor --> tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        assert tokenizer.pad_token_id is not None
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
    
    @abstractmethod
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        ...