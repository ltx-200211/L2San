"""Original operating-point policy; s is a preference, not a DP budget."""
import math
BUDGET = {0:9, .25:9, .45:9, .5:9, .525:9, .55:7, .625:7, .74:5}


def configuration(s):
    if not math.isfinite(s) or not 0 <= s <= 1: raise ValueError('s must lie in [0,1]')
    if s < .75 and s not in BUDGET:
        raise ValueError('Old policy has discrete low-s budget anchors; use '+str(sorted(BUDGET)))
    return dict(s=s, epsilon=BUDGET[s] if s < .75 else (0 if s == 1 else .1*1000**(1-s)),
                k=2 if s < .25 else 4)


def decide(probabilities, subtypes, s):
    """probabilities: group->(p1,p23,p4); subtypes: entity->(group,M2/M3)."""
    configuration(s)
    if not probabilities: raise ValueError('Missing group probabilities')
    for p in probabilities.values():
        if len(p)!=3 or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or not math.isclose(sum(p),1.,abs_tol=1e-4):
            raise ValueError('Invalid routing probabilities')
    bias = 4*math.log((1-s)/s) if 0<s<1 else 0
    if s == 0 or (s < 1 and any(math.log(max(p[2],1e-12))+bias > math.log(max(p[0],p[1],1e-12)) for p in probabilities.values())):
        return {eid:'M4' for eid in subtypes}
    out = {}
    for eid,(group,subtype) in subtypes.items():
        if subtype not in ('M2','M3'): raise ValueError('M23 requires a resolved M2/M3 subtype')
        p = probabilities[group]
        out[eid] = 'M1' if s<.75 and p[0]>=p[1] else subtype if s<.9 else 'M3'
    return out
