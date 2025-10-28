from typing import Dict, List
import re

from larm.data.envs.base_env import StaticEnv
from larm.common.registry import registry
from larm.memory_generator.trainer.verifier import verify_solution_equivalence
from larm.memory_generator.trainer.utils import extract_answer


@registry.register_env("mmvp")
class MMVPEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def _accuracy_reward(cls, completions: List[str], solution: List[str], **kwargs) -> List[float]:

        scores: List[float] = []
        for c, s in zip(completions, solution):
            candidate = extract_answer(c)
            try:
                # # 使用正则匹配提取选项字母，支持 (a), (b), a, b 等格式
                # candidate_match = re.search(r'\(?([a-d])\)?', candidate.lower()) if candidate else None
                # solution_match = re.search(r'\(?([a-d])\)?', s.lower()) if s else None
                # ok = (candidate_match and solution_match and 
                #       candidate_match.group(1) == solution_match.group(1))
                ok = verify_solution_equivalence(candidate, s)
            except Exception:
                ok = False
            scores.append(1.0 if ok else 0.0)
        return scores

    @classmethod
    def _format_reward(cls, completions: List[str], **kwargs):
        pass
