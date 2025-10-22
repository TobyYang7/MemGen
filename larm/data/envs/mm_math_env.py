from typing import Dict, List

from larm.data.utils.math_utils import compute_score
from larm.data.envs.base_env import StaticEnv
from larm.common.registry import registry


@registry.register_env("mm_math")
class MMMathEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def _accuracy_reward(cls, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = [compute_score(completion=c, ground_truth=s) for c, s in zip(completions, solution)]
        return scores

    @classmethod
    def _format_reward(cls, completions: List[str], **kwargs):
        pass
