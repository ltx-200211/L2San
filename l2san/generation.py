from . import expansion as s
d=s.d;h=s.h

def generate_seed(case,e,lc,cid):
    plan=d.mask_plan(case,e);mask,(left,right),typ=d.context_from_plan(case,e,plan,'target')
    ordinary=d.ordinary(mask,left,right,typ,lc,cid,e['id'],'target');ordinary['masked_context']=mask
    branches,evidence=d.diverse_lists(mask,left,right,typ,lc,cid,e['id'])
    branches={'context':ordinary['candidates'],**branches}
    all_cs=list(dict.fromkeys(c for vs in branches.values() for c in vs));logs=dict(ordinary['logprobs'])
    missing=[c for c in all_cs if c not in logs]
    if missing:logs.update(h.public_logprobs(mask,left,right,missing,lc))
    seed=d.merge_branches(branches,logs) or [f'⟦PUBLIC_{typ}⟧']
    return {'id':cid,'entity_id':e['id'],'type':typ,'mask_plan':plan,'records':{'target':ordinary},'branches':branches,'list_generation_evidence':evidence,'supports':{'target_diverse':seed},'logprobs':logs,'static_bank_used':False}

def expand_seed(src,lc,seed=d.SEED):
    cid,eid=src['id'],src['entity_id'];mask=src['records']['target']['masked_context'];left,right=mask.split('[TARGET]');typ=src['type']
    initial=src['supports']['target_diverse'];branches={k:list(src['branches'][k]) for k in s.PROMPTS};evidence=[]
    # Keep the original expansion horizon (192), then take its dynamic96 prefix.
    # Stopping generation at 96 would change logprob-ranked support selection.
    for round_id in range(4):
        for branch,prompt in s.PROMPTS.items():
            known=list(dict.fromkeys(initial+[c for v in branches.values() for c in v]))
            if len({h.q.norm(c) for c in known})>=192:break
            for attempt in range(3):
                try:
                    choice=d.ask(prompt+'\n'+s.EXTRA,{'masked_task':mask,'target_type':typ,'left':left[-180:],'right':right[:180],'already_generated':known},lc,1,3000,h.q.stable_seed(seed,'poolscale',cid,eid,branch,round_id,attempt))['choices'][0]
                    assert choice.get('finish_reason')!='length','truncated candidate list'
                    tokenlogs=[float(t['logprob']) for t in (choice.get('logprobs') or {}).get('content',[]) if t.get('logprob') is not None];assert tokenlogs
                    raw=d.parse_json(choice['message']['content'])['candidates'];assert isinstance(raw,list)
                    seen={h.q.norm(c) for c in known};fresh=[]
                    for c in raw:
                        c=h.clean(c) if isinstance(c,str) else None
                        if c and '[TARGET_COPY]' not in c and h.q.norm(c)!=h.q.norm(typ) and h.q.norm(c) not in seen:
                            fresh.append(c);seen.add(h.q.norm(c))
                    valid=d.gate(mask,left,right,typ,fresh,lc,d.ROLE);branches[branch].extend(valid)
                    evidence.append({'branch':branch,'round':round_id,'raw':raw,'new_before_gate':len(fresh),'kept':valid,'generation_tokens':len(tokenlogs),'mean_generation_token_logprob':sum(tokenlogs)/len(tokenlogs)})
                    break
                except Exception as exc:
                    if attempt==2:evidence.append({'branch':branch,'round':round_id,'raw':[],'new_before_gate':0,'kept':[],'generation_tokens':0,'error':repr(exc),'status':'skipped_after_three_failed_attempts'})
        if len({h.q.norm(c) for c in initial+[c for v in branches.values() for c in v]})>=192:break
    allcs=list(dict.fromkeys(initial+[c for v in branches.values() for c in v]));logs=dict(src['logprobs']);missing=[c for c in allcs if c not in logs]
    if missing:logs.update(h.public_logprobs(mask,left,right,missing,lc))
    supports=s.nested(initial,branches,logs)
    assert supports['dynamic24']==initial and 0<len(supports['dynamic96'])<=96
    return {'id':cid,'entity_id':eid,'type':typ,'mask_plan':src['mask_plan'],'masked_context':mask,'supports':supports,'branches':branches,'list_generation_evidence':evidence,'dictionary':[],'generated':supports['dynamic192'],'public_fallback':None,'logprobs':{c:logs[c] for c in supports['dynamic192'] if c in logs},'static_bank_used':False,'source':'same target-only seed and nested expansion as m3_dynamic_poolscale20_20261008'}
