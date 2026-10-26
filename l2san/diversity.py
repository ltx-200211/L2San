import re, json
from . import proposals as h
SEED = 20260925

ALT='''Generate 16 semantically DIFFERENT possible replacements for TARGET.
Use the target type, grammar, and visible context. The hidden original is unknown.
Explore distinct plausible meanings, not just paraphrases, spelling variants or
small intensity changes of one likely guess. Do not restrict the list to the most
likely reconstruction. Preserve the slot's role (e.g. observation versus diagnosis,
legal action versus document, a clause versus a noun), tense and grammatical form.
Do not answer or solve the question, copy its answer choices as an answer, or give
advice. These are possible substitutes, not assertions about the hidden original.
Output JSON only: {"candidates":["standalone replacement", "..."]}.'''

BROAD='''Generate 16 usable broader replacements for TARGET at different levels of
specificity. Only the supplied type and visible context are known. Include natural
category phrases, less specific descriptions and short clauses when grammatically
appropriate. Express varied broad meanings, not many synonyms of one phrase. Keep
useful information about the slot's role without inventing a diagnosis, reason,
legal conclusion, or concrete fact not supported by the visible context. Avoid
empty placeholders such as something, TARGET or a type ID. Do not solve the task.
Output JSON only: {"candidates":["standalone replacement", "..."]}.'''

ROLE='''Check possible replacements for TARGET using only the supplied masked query.
Keep candidates that can occupy the slot with its grammatical and semantic role.
They are alternative possible meanings, not necessarily the likeliest hidden value.
Allow broader but meaningful category phrases and clauses. Do not discard an item
merely because another is more probable. Reject task answers, advice, lists, broken
fragments, repetitions of surrounding text, and placeholders. Do not infer or score
agreement with the hidden original. JSON only: {"keep":[0,1,...]}.'''

def mask_plan(case, entity):
    """Fixed reviewed equivalences, never rematch private strings in context."""
    members = [e for e in case['entities'] if e.get('canonical_entity_id', e['id']) == entity['id']]
    assert members
    return {'target_span': [entity['start'], entity['end']],
            'hidden_spans': [[e['start'], e['end']] for e in sorted(members, key=lambda e:e['start'])],
            'linked_ids': [e['id'] for e in members], 'canonical_entity_id': entity['id'],
            'link_source': 'explicit_equivalence_v1'}

def context_from_plan(case,e,plan,mode):
    ranges=[(a,b,'target' if a<=e['start'] and b>=e['end'] else 'copy') for a,b in plan['hidden_spans']]
    if mode=='all':
        for ent in case['entities']:
            if any(ent['start']<b and ent['end']>a for a,b,_ in ranges):continue
            ranges.append((ent['start'],ent['end'],'other:'+ent.get('operational_type',ent['type'])+'_'+ent['id']))
    parts=[];last=0
    for a,b,kind in sorted(ranges):
        assert a>=last
        marker='[TARGET]' if kind=='target' else '[TARGET_COPY]' if kind=='copy' else '⟦'+kind[6:]+'⟧'
        parts.extend([case['query'][last:a],marker]);last=b
    parts.append(case['query'][last:]);masked=''.join(parts)
    assert masked.count('[TARGET]')==1
    return masked,masked.split('[TARGET]'),e.get('operational_type',e['type'])

def parse_json(s):
    start=s.find('{');end=s.rfind('}');return json.loads(s[start:end+1])

def ask(system,payload,lc,n,max_tokens,seed):
    return h.q.chat_completion_request({'model':lc.local_model,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
        'n':n,'max_tokens':max_tokens,'temperature':1.0,'top_p':1.0,'logprobs':True,'seed':seed%(2**31),'chat_template_kwargs':{'enable_thinking':False}},lc)

def gate(mask,left,right,typ,values,lc,role):
    kept=[]
    for start in range(0,len(values),30):
        cs=values[start:start+30]
        for attempt in range(3):
            try:
                raw=h.q.local_json(role,{'masked_task':mask,'target_type':typ,'left':left[-180:],'right':right[:180],
                    'candidates':[{'index':i,'text':c} for i,c in enumerate(cs)]},lc,700,temperature=0)
                indices=raw['keep'];assert all(type(i) is int and 0<=i<len(cs) for i in indices)
                kept.extend(cs[i] for i in dict.fromkeys(indices));break
            except Exception:
                if attempt==2:raise
    return kept

def ordinary(mask,left,right,typ,lc,cid,eid,mode):
    values={};details={};tokens=0
    # Same sampling cap as the prior ordinary generator; only context varies.
    for attempt in range(3):
        prompt=h.GEN if mode=='all' else h.GEN.replace('whose protected spans are masked','whose TARGET and linked copies are masked')
        raw=ask(prompt,{'masked_task':mask,'target_type':typ,'immediate_left':left[-180:],'immediate_right':right[:180],
            'instruction':'Output only the replacement occupying TARGET, not the surrounding text.'},lc,32,96,h.q.stable_seed(SEED,'hybrid-public-generation',cid,eid,attempt))
        for choice in raw['choices']:
            if choice.get('finish_reason')=='length':continue
            c=h.clean(choice['message']['content'])
            if not c or '[TARGET_COPY]' in c or h.q.norm(c)==h.q.norm(typ):continue
            logs=[float(t['logprob']) for t in (choice.get('logprobs') or {}).get('content',[]) if t.get('logprob') is not None and not t.get('token','').startswith('<|')]
            if not logs:raise ValueError('Missing actual generation token logprobs')
            tokens+=len(logs);key=h.q.norm(c)
            if key not in values or sum(logs)/len(logs)>details[values[key]]['generation_logprob']:
                values[key]=c;details[c]={'generation_logprob':sum(logs)/len(logs),'generation_tokens':len(logs)}
        valid=gate(mask,left,right,typ,list(values.values()),lc,h.ROLE)
        values={h.q.norm(c):c for c in valid}
        if len(values)>=24:break
    cs=list(values.values())
    logs=h.public_logprobs(mask,left,right,cs,lc) if cs else {}
    cs.sort(key=lambda c:(-logs[c]['public_logprob'],h.q.norm(c)));cs=cs[:24]
    return {'candidates':cs,'logprobs':{c:logs[c] for c in cs},'generation_details':{c:details[c] for c in cs},'generation_tokens_total':tokens}

def diverse_lists(mask,left,right,typ,lc,cid,eid):
    branches={};evidence={}
    for branch,prompt in [('alternative',ALT),('broad',BROAD)]:
        for attempt in range(3):
            try:
                choice=ask(prompt,{'masked_task':mask,'target_type':typ,'left':left[-180:],'right':right[:180]},lc,1,1800,
                    h.q.stable_seed(SEED,'semantic-diverse-list',cid,eid,branch,attempt))['choices'][0]
                assert choice.get('finish_reason')!='length'
                logs=[float(t['logprob']) for t in (choice.get('logprobs') or {}).get('content',[]) if t.get('logprob') is not None]
                assert logs,'Missing actual list generation logprobs'
                raw=parse_json(choice['message']['content']);assert isinstance(raw['candidates'],list)
                vals={}
                for c in raw['candidates']:
                    c=h.clean(c) if isinstance(c,str) else None
                    if c and '[TARGET_COPY]' not in c and h.q.norm(c)!=h.q.norm(typ):vals.setdefault(h.q.norm(c),c)
                valid=gate(mask,left,right,typ,list(vals.values())[:24],lc,ROLE)
                branches[branch]=valid;evidence[branch]={'raw_candidates':list(vals.values()),'generation_tokens':len(logs),'mean_generation_token_logprob':sum(logs)/len(logs),'attempt':attempt+1}
                break
            except Exception:
                if attempt==2:raise
    return branches,evidence

def merge_branches(branches,logs,cap=24):
    # Equal source quotas encourage coverage, without private-value selection.
    ordered={k:sorted(v,key=lambda c:(-logs[c]['public_logprob'],h.q.norm(c))) for k,v in branches.items()}
    result={}
    for name in ['context','alternative','broad']:
        for c in ordered[name][:8]:result.setdefault(h.q.norm(c),c)
    # Fill unused slots only from already generated public-context candidates.
    remaining=sorted({c for vs in ordered.values() for c in vs},key=lambda c:(-logs[c]['public_logprob'],h.q.norm(c)))
    for c in remaining:
        if len(result)>=cap:break
        result.setdefault(h.q.norm(c),c)
    return list(result.values())[:cap]
