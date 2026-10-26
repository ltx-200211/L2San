"""Run method inference only. Public and private records are separate files."""
import argparse, json
from pathlib import Path
from types import SimpleNamespace
from . import routing, m2, m3, m4, client, subtyping
ROOT = Path(__file__).resolve().parent.parent


def read(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def router_input(case):
    return dict(query_id=case['id'],query=case['query'],entities=case['entities'],
        groups=[dict(id=g['group_id'],entities=g['entity_ids']) for g in case['groups']])


def render(case, replacements):
    parts=[];cursor=0
    for e in sorted(case['entities'], key=lambda e:e['start']):
        start,end=e['start'],e['end']
        if start<cursor or case['query'][start:end]!=e['value']:
            raise ValueError('Invalid or overlapping annotated spans')
        parts.extend([case['query'][cursor:start],replacements[e.get('canonical_entity_id',e['id'])]])
        cursor=end
    return ''.join(parts)+case['query'][cursor:]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['router-input','sanitize'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predictions',type=Path);p.add_argument('--s',type=float,default=.5)
    p.add_argument('--local-url',default='http://127.0.0.1:8001/v1');p.add_argument('--local-model',default='privacy-qwen')
    p.add_argument('--m4-url',default='http://127.0.0.1:18008/v1');p.add_argument('--m4-model',default='privacy-qwen35-4b')
    p.add_argument('--seed',type=int,default=20261024);p.add_argument('--score-workers',type=int,default=8)
    a=p.parse_args();cases=read(a.input)
    if a.output.exists():raise ValueError('Use a fresh output path')
    if a.command=='router-input':
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(''.join(json.dumps(router_input(c),ensure_ascii=False)+'\n' for c in cases));return
    if not a.predictions:p.error('--predictions is required')
    cfg=routing.configuration(a.s);preds={r['query_id']:r for r in read(a.predictions)}
    local=SimpleNamespace(local_url=a.local_url,local_model=a.local_model,timeout=300)
    m4local=SimpleNamespace(local_url=a.m4_url,local_model=a.m4_model,timeout=300)
    broad=json.loads((ROOT/'configs/broad32.json').read_text())['candidates']
    policies=json.loads((ROOT/'configs/m2_policies.json').read_text())
    a.output.mkdir(parents=True);(a.output/'private').mkdir(mode=0o700)
    for case in cases:
        cid=case['id']
        if Path(cid).name!=cid:raise ValueError('Unsafe query id')
        pred=preds[cid]
        if pred['class_order']!=['C1','C23','C4'] or len(pred['probabilities'])!=len(case['groups']):raise ValueError('Router schema mismatch')
        probs={g['group_id']:p for g,p in zip(case['groups'],pred['probabilities'])}
        # M4 and all-M3 endpoints do not require a subtype call.
        provisional={e['id']:(e['semantic_group_id'],'M3') for e in case['units']}
        routes=routing.decide(probs,provisional,a.s)
        private={'configuration':cfg,'routes':routes}
        if 'M4' in routes.values():
            public,truth=m4.generate(case,m4local,cfg['k'],client.stable_seed(a.seed,cid,'m4'))
            private.update(truth)
        else:
            if a.s<.9:
                subtype=subtyping.classify(case,local)
                routes=routing.decide(probs,{e['id']:(e['semantic_group_id'],subtype[e['id']]) for e in case['units']},a.s)
                private['routes']=routes
            replacements={};private['banks']={}
            for e in case['units']:
                eid=e['id'];route=routes[eid]
                if route=='M1':val='[PRIVATE]'
                elif route=='M2':
                    val,_=m2.replace_all(case,dict(e,value=subtyping.numeric_text(e)),cfg['epsilon'],policies,client.stable_seed(a.seed,'m2',cid,eid))
                else:
                    if a.s>=.9:values={c:0. for c in broad}
                    else:
                        bank=m3.generate(case,e,local)
                        # Match historical main: score the full dynamic96 support, then public subset32.
                        scores=m3.score(case,e,bank['candidates'],local,a.score_workers)
                        private['banks'][eid]=dict(bank,scores=scores)
                        values=m3.support_for_strength(bank['candidates'],scores,broad,a.s)
                    val=m3.replace(values,cfg['epsilon'],client.stable_seed(a.seed,'uniform-ab',cid,eid))
                replacements[eid]=val
            public={'text':render(case,replacements)};private['replacements']=replacements
        dump(a.output/'public'/f'{cid}.json',public)
        dump(a.output/'private'/f'{cid}.json',private)
        print(cid,flush=True)

if __name__=='__main__':main()
