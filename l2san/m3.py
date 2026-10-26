"""Masked generation -> public logprob ranking -> fixed support -> private scores."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from . import diversity as d, expansion as x, scoring, sampling


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def subset(candidates, cap=32):
    if len(candidates) != len(set(candidates)) or not candidates:
        raise ValueError('Candidate support must be nonempty and unique')
    selected = set(sorted(candidates, key=lambda c: (digest([2026102101, c]), c))[:cap])
    return [c for c in candidates if c in selected]


def generate(case, entity, local, seed=20260925):
    from .generation import generate_seed, expand_seed
    bank = expand_seed(generate_seed(case, entity, local, case['id']), local, seed)
    support = bank['supports']['dynamic96']
    if len(support) < 32: raise ValueError('Fewer than 32 candidates; preparation failed')
    return {'candidates': support, 'mask_plan': bank['mask_plan'], 'masked_context': bank['masked_context']}



def score(case, entity, candidates, local, workers=8):
    # Scoring must NEVER remove items from the already frozen support.
    zero, _ = scoring.screen(case, entity, candidates, local, 48, 'keep')
    remaining = [c for i, c in enumerate(candidates) if i not in zero]
    def one(c): return scoring.fine_scores(case, entity, [c], local)[0]
    components = {c: {'semantic':0., 'task':0.} for i,c in enumerate(candidates) if i in zero}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(one, remaining): components.update(result)
    return {c: .8*components[c]['semantic'] + .2*components[c]['task'] for c in candidates}


def support_for_strength(candidates, values, broad, strength):
    if strength >= .9: return {c: 0. for c in broad}
    context = subset(candidates, 32)
    chosen = context if strength < .75 else context[:16]
    result = {c: values[c] for c in chosen}
    if strength >= .75:
        seen = {d.h.q.norm(c) for c in result}
        for c in broad:
            if d.h.q.norm(c) not in seen:
                result[c] = 0.; seen.add(d.h.q.norm(c))
            if len(result) == 32: break
    if len(result) != 32: raise ValueError('Insufficient distinct public candidates')
    return result


def replace(values, epsilon, seed):
    return sampling.draw(sampling.probabilities(values, epsilon, 'winner_rr'), seed)
