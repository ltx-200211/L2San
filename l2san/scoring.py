import json, math, time
from . import diversity as d
h = d.h

ZERO='''Perform a conservative, coarse screen of replacement candidates for ONE PRIVATE target.
Return ONLY indices that clearly deserve BOTH semantic=0 AND task=0.
Focus on original_target and its marked location, NOT on other entities in the question.
Zero candidates that reverse its explicit negation, actor/object relation or factual state;
replace an observation with an unsupported diagnosis/conclusion; or express an unrelated
fact that loses the original clue and would mislead the task even after local name/reference restoration.
A word occurring elsewhere in the question is not evidence of target equivalence.
KEEP exact equivalents, potentially useful truthful generalizations, and uncertain cases
for later fine scoring. Different surface names may be restored locally: do not zero them
solely for being different names. Mere low similarity is insufficient to assign zero.
Do not solve the question. Do not give fine scores. Output JSON only:
{"zero":[0,3,8]}. Every omitted index goes to fine scoring. An empty list is valid.'''

KEEP='Select every candidate that MAY deserve any nonzero semantic or task-preservation score for the PRIVATE original target. This is a high-recall shortlist, not a ranking or a search for the best candidate.\nCompare ONLY the marked original_target with each possible replacement in its slot. Select exact matches, paraphrases, related meanings, plausible generalizations, same-role names that local reference restoration could recover, and ALL uncertain or partly useful candidates. Preserve negation and actor/object roles. Select even a weakly related candidate if its usefulness is uncertain. Reject only clearly irrelevant or contradictory changes that lose the target\'s clue entirely. A candidate merely occurring elsewhere in the question does not make it equivalent to the target.\nSynthetic example: for \'declined the invitation\', keep \'did not accept the invitation\' and \'was unwilling to participate\'.\nSynthetic example: for \'has no access to the account\', \'can access the account\' reverses the fact. A broader phrase expressing access difficulties can still be retained for fine scoring.\nReturn JSON only: {"keep":[0,1,3]}. Include ALL possibly useful indices, not just a few best ones.'

def payload(case,e,cs):
    q=case['query'];marked=q[:e['start']]+'<PRIVATE_TARGET>'+q[e['start']:e['end']]+'</PRIVATE_TARGET>'+q[e['end']:]
    return {'original_task':marked,'original_target':e['value'],'target_type':e.get('operational_type',e['type']),
            'required_ids':list(range(len(cs))),'candidates':[{'id':i,'text':c} for i,c in enumerate(cs)]}

def request_json(prompt,body,lc,max_tokens,events,validate):
    for attempt in range(3):
        start=time.perf_counter();ev={'attempt':attempt+1}
        try:
            raw=h.q.chat_completion_request({'model':lc.local_model,'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(body,ensure_ascii=False)}],
              'temperature':0,'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':False}},lc)
            ev.update(raw.get('usage') or {});choice=raw['choices'][0]
            assert choice.get('finish_reason')!='length','truncated JSON'
            obj=d.parse_json(choice['message']['content']);validate(obj);return obj
        except Exception as exc:
            ev['error']=repr(exc)
            if attempt==2:raise
        finally:ev['seconds']=time.perf_counter()-start;events.append(ev)

def zero_ids(obj,n):
    ids=obj['zero']
    if not isinstance(ids,list) or any(type(i) is not int or not 0<=i<n for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Invalid zero indices')
    return set(ids)

def screen(case,e,cs,lc,batch_size=48,screen_mode="keep"):
    rejected=set();events=[];failures=[];identity_overrides=[]
    for start in range(0,len(cs),batch_size):
        chunk=cs[start:start+batch_size]
        try:
            field='zero' if screen_mode=='zero' else 'keep'
            obj=request_json(ZERO if screen_mode=='zero' else KEEP,payload(case,e,chunk),lc,900,events,lambda o:zero_ids({'zero':o[field]},len(chunk)))
            selected=zero_ids({'zero':obj[field]},len(chunk))
            rejected_chunk=selected if screen_mode=='zero' else set(range(len(chunk)))-selected
            for i in rejected_chunk:
                if h.q.norm(chunk[i])==h.q.norm(e['value']):identity_overrides.append(start+i)
                else:rejected.add(start+i)
        except Exception as exc:
            # Parse/transport failure never silently zeroes a candidate.
            failures.append({'start':start,'error':repr(exc)})
    return rejected,{'calls':events,'fallback_batches':failures,'identity_overrides':identity_overrides}

def fine_scores(case,e,cs,lc):
    components={};events=[]
    for start in range(0,len(cs),1):
        chunk=cs[start:start+1];body=payload(case,e,chunk)
        body['output_requirement']=f'Return exactly {len(chunk)} separate score objects. Include every required ID even when scores are identical. Do not combine or omit candidates.'
        def validate(obj):
            raw=obj['scores'];assert len(raw)==len(chunk)
            by={v['id']:v for v in raw};assert set(by)==set(range(len(chunk)))
            for v in by.values():
                for k in ['semantic','task']:assert math.isfinite(float(v[k])) and 0<=float(v[k])<=100
        obj=request_json(h.SCORE,body,lc,1300,events,validate)
        by={v['id']:v for v in obj['scores']}
        for i,c in enumerate(chunk):
            components[c]={k:float(by[i][k])/100 for k in ['semantic','task']}
            if h.q.norm(c)==h.q.norm(e['value']):components[c]={'semantic':1.,'task':1.}
    return components,events

def utility(comp):return .8*comp['semantic']+.2*comp['task']
