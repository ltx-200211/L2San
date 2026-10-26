"""Same-task parallel requests, uniform style, fidelity and null-question guard."""
import json, random, re
from . import client, m4_prompts, style_prompts
NULL_QUESTION = r'Question:\s*(?:none|n/?a|not specified|no (?:explicit )?questions?(?: (?:stated|asked))?)\.?\s*$'


def obj(props):
    return dict(type='object', properties=props, required=list(props), additionalProperties=False)


def array(item,n): return dict(type='array', minItems=n,maxItems=n,items=item)


def call(prompt, payload, schema, local, **kwargs):
    request = dict(model=local.local_model, messages=[dict(role='system',content=prompt),
        dict(role='user',content=json.dumps(payload,ensure_ascii=False))], temperature=.2,
        top_p=.95, max_tokens=4000, seed=20261025, chat_template_kwargs={'enable_thinking':False},
        structured_outputs={'json':schema})
    request.update(kwargs)
    raw = client.chat_completion_request(request, local)['choices'][0]
    if raw.get('finish_reason')=='length': raise ValueError('Truncated M4 generation')
    return json.loads(raw['message']['content'])


def generate(case, local, k=4, seed=20261026):
    if k not in (2,4): raise ValueError('Released policy supports k=2 or k=4')
    # Generate the same quartet first. k=2 is its true request plus one peer.
    schema=obj(dict(peers=array({'type':'string'},3)))
    for attempt in range(2):
        try:
            generated=call(m4_prompts.PROMPT, dict(authentic_request=case['query'],
                protected_facts=[dict(id=e['id'],value=e['value'],type=e.get('operational_type',e['type'])) for e in case['units']]),
                schema,local, temperature=.7,top_p=.95,top_k=20,presence_penalty=.4,
                max_tokens=4096 if attempt==0 else 6144,seed=20261026+attempt,
                chat_template_kwargs={'enable_thinking':True},thinking_token_budget=1024)
            peers=generated['peers']
            if len(peers)!=3 or any(not isinstance(t,str) or len(t.strip())<=30 for t in peers): raise ValueError('Invalid peers')
            original=[case['query']]+[t.strip() for t in peers]
            if len(set(original))!=4: raise ValueError('Duplicate requests')
            break
        except Exception:
            if attempt: raise
    rng=random.Random(seed); order=list(range(4));rng.shuffle(order)
    parallel=[original[i] for i in order];truth=order.index(0);texts=parallel;normalized=False
    try:
        schema=obj(dict(requests=array(obj(dict(facts={'type':'array','minItems':1,'items':{'type':'string'}},question={'type':'string'})),4)))
        norm=call(style_prompts.PROMPT, {'queries':[dict(index=i,text=t) for i,t in enumerate(parallel)]}, schema,local)
        if len(norm['requests'])!=4: raise ValueError('Missing normalized request')
        rewritten=['Facts:\n'+'\n'.join('- '+f.strip() for f in r['facts'])+'\nQuestion: '+r['question'].strip() for r in norm['requests']]
        if len(set(rewritten))!=4 or any(len(t)<=30 for t in rewritten):raise ValueError('Invalid normalized requests')
        checks=call(style_prompts.VERIFY,dict(pairs=[dict(original=o,rewritten=t) for o,t in zip(parallel,rewritten)]),
            obj(dict(checks=array(obj(dict(faithful={'type':'boolean'},reason={'type':'string'})),4))),local)['checks']
        if len(checks)==4 and all(c['faithful'] is True for c in checks) and not any(re.search(NULL_QUESTION,t,re.I) for t in rewritten):
            texts=rewritten; normalized=True
    except Exception:
        pass  # whole-quartet fallback, never choose by external attack outcome
    if k==2:
        peer=rng.choice([i for i in range(4) if i!=truth]);keep=[truth,peer];rng.shuffle(keep)
        texts=[texts[i] for i in keep];truth=keep.index(truth)
    return {'queries':[dict(index=i,text=t) for i,t in enumerate(texts)]}, {'true_index':truth,'normalized':normalized}
