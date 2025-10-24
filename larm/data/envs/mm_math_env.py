from typing import Dict, List

from larm.data.envs.base_env import StaticEnv
from larm.common.registry import registry
from larm.memory_generator.trainer.verifier import verify_solution_equivalence


@registry.register_env("mm_math")
class MMMathEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def _accuracy_reward(cls, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        def _extract_answer(text: str) -> str:
            try:
                low = text.lower()
                s = low.find("<answer>")
                e = low.find("</answer>")
                if s != -1 and e != -1 and e > s:
                    return text[s + len("<answer>") : e].strip()
            except Exception:
                pass
            return ""

        scores: List[float] = []
        for c, s in zip(completions, solution):
            candidate = _extract_answer(c)
            try:
                ok = verify_solution_equivalence(candidate, s)
            except Exception:
                ok = False
            scores.append(1.0 if ok else 0.0)
        return scores

    @classmethod
    def _format_reward(cls, completions: List[str], **kwargs):
        pass
