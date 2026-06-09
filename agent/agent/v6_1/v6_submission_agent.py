"""
Wrapper opponent that delegates to the packaged v6 submission agent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V6_SUBMISSION_DIR = _HERE.parent / "v6" / "v6_submission"
_V6_SUBMISSION_AGENT = _V6_SUBMISSION_DIR / "agent.py"


def _load_v6_submission_agent_class():
    module_name = "_v6_submission_eval_agent"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, _V6_SUBMISSION_AGENT)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load v6 submission agent from {_V6_SUBMISSION_AGENT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        raise AttributeError(f"No Agent class found in {_V6_SUBMISSION_AGENT}")
    return agent_cls


class V6SubmissionAgent:
    def __init__(self, agent_id: int):
        agent_cls = _load_v6_submission_agent_class()
        self._delegate = agent_cls(agent_id)

    def act(self, obs):
        return int(self._delegate.act(obs))
