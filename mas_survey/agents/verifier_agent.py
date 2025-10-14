# mas_survey/agents/verifier_agent.py
from decimal import Decimal, getcontext
getcontext().prec = 50

def calibrate_distribution(best_symbol: str, option_list: list[str]) -> dict[str, float]:
    """
    Deterministic calibration:
    - 0.7 on the chosen option, rest spread evenly on others.
    - Final correction to make sum exactly 1.0 (±1e-6).
    """
    if not option_list:
        return {"A": 1.0}
    if best_symbol not in option_list:
        # fallback: uniform
        n = len(option_list)
        vals = [float(Decimal(1)/Decimal(n)) for _ in range(n)]
        vals[-1] += (1.0 - sum(vals))
        return {option_list[i]: vals[i] for i in range(n)}

    base = Decimal("0.7")
    remain = Decimal("1.0") - base
    others = [o for o in option_list if o != best_symbol]
    if not others:
        return {best_symbol: 1.0}

    each = remain / Decimal(len(others))
    bag = {o: float(each) for o in others}
    bag[best_symbol] = float(base)

    # exact sum correction
    keys = list(bag.keys())
    diff = 1.0 - sum(bag.values())
    bag[keys[-1]] += diff
    return bag

def enforce_unique_supports(candidates: list[str], want_k: int, universe_ids: list[str]) -> list[str]:
    """
    Deduplicate candidates; if fewer than K, pad from universe_ids (stable order).
    Guarantees exactly K unique IDs (assuming universe has enough).
    """
    out, seen = [], set()
    for cid in candidates:
        if cid not in seen:
            out.append(cid); seen.add(cid)
        if len(out) >= want_k:
            return out[:want_k]
    for uid in universe_ids:
        if uid not in seen:
            out.append(uid); seen.add(uid)
        if len(out) >= want_k:
            break
    return out[:want_k]
