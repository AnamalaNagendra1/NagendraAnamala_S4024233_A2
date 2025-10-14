# mas_survey/agents/parser_agent.py
def parse_question(text: str):
    """
    Minimal parser. Returns a normalized question and default options (A–D).
    If your input JSON already has options, the runner will use those instead.
    """
    return {
        "qtype": "mcq",
        "options": ["A", "B", "C", "D"],
        "normalized": (text or "").strip()
    }
