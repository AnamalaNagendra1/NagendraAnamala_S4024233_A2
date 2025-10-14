# mas_survey/agents/responder_agent.py
from collections import Counter
import re

_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")

def _toks(s: str):
    return [t.lower() for t in _TOKEN_RX.findall(s or "")]

def pick_answer(normalized_question: str, options: list[str]) -> str:
    """
    Deterministic placeholder:
    - If an option token appears literally in the question, pick it.
    - Otherwise, return the first option.
    """
    bag = Counter(_toks(normalized_question))
    for opt in options or []:
        if opt.lower() in bag:
            return opt
    return options[0] if options else "A"
