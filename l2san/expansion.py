from . import diversity as d
h = d.h
CAPS = {"dynamic24":24,"dynamic96":96,"dynamic192":192}

PROMPTS={
 'context':h.GEN.replace('Return one plausible replacement text','Generate 32 different plausible replacement texts').replace('Output the replacement text only, without quotation marks.','Return JSON only: {"candidates":["replacement", "..."]}.'),
 'alternative':d.ALT.replace('16','32'),
 'broad':d.BROAD.replace('16','32')}

EXTRA='''The supplied already_generated list contains earlier proposals. Generate NEW meanings absent from that list; avoid synonymous or superficial variations. Explore different plausible circumstances within the SAME target type and slot role. Do not force 32 if fewer coherent alternatives exist. The target and its copies are hidden; other visible context is available. Never reconstruct a placeholder.'''

def nested(seed, branches, logs):
    result=list(seed);seen={h.q.norm(c) for c in result}
    ordered={k:sorted(v,key=lambda c:(-logs[c]['public_logprob'],h.q.norm(c))) for k,v in branches.items()}
    for i in range(max([len(v) for v in ordered.values()]+[0])):
        for name in PROMPTS:
            if i<len(ordered[name]):
                c=ordered[name][i];key=h.q.norm(c)
                if key not in seen:result.append(c);seen.add(key)
    return {arm:(list(seed) if cap==24 else result[:cap]) for arm,cap in CAPS.items()}
