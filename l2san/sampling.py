import math, random
from types import SimpleNamespace
from . import client
h = SimpleNamespace(q=client)
ARMS = ["utility_em", "rank_power4", "winner_rr"]

def probabilities(values,epsilon,arm):
    if not values or not math.isfinite(epsilon) or epsilon<0:raise ValueError('Invalid support or epsilon')
    if any(not math.isfinite(u) or not 0<=u<=1 for u in values.values()):raise ValueError('Unbounded score')
    cs=list(values);n=len(cs)
    if arm=='utility_em':
        w=[math.exp(epsilon*u/2) for u in values.values()];z=sum(w)
    else:
        if arm not in ARMS:raise ValueError(arm)
        profile=[((n-1-i)/max(1,n-1))**4 for i in range(n)] if arm=='rank_power4' else [1.]+[0.]*(n-1)
        # Public score multiset, uniform base measure: Z is independent of x.
        rank_weights=[math.exp(epsilon*a) for a in profile];z=sum(rank_weights)
        order=sorted(range(n),key=lambda i:(-round(values[cs[i]],12),i));w=[0.]*n;start=0
        while start<n:
            end=start+1
            while end<n and round(values[cs[order[end]]],12)==round(values[cs[order[start]]],12):end+=1
            # Average WEIGHTS, not scores, within each private tie group.
            # This preserves the public Z exactly and min/max weight bounds.
            mean=sum(rank_weights[start:end])/(end-start)
            for i in order[start:end]:w[i]=mean
            start=end
        assert math.isclose(sum(w),z,rel_tol=1e-12)
    return {c:weight/z for c,weight in zip(cs,w)}

def draw(probabilities,seed):
    def objective(c):
        u=random.Random(h.q.stable_seed(seed,'candidate-gumbel',c)).random()
        return math.log(probabilities[c])-math.log(-math.log(max(u,1e-300))),c
    return max(probabilities,key=objective)
