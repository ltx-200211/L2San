"""All-coordinate metric perturbation, conditional on a fixed public text template.

Only numeric magnitudes are private in this mechanism's metric claim. Token
count, units, format and coordinate policies must be held fixed across neighbors.
The private-dependent router is outside that conditional claim.
"""
import calendar
import math
import random
import re
import secrets

DIGITS = re.compile(r'(?:(?<![\w.])[+-])?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|(?<!\w)[+-]?\.\d+')
WORDS = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,
         'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,
         'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,
         'hundred':100,'thousand':1000,'million':1000000,'half':0.5,'quarter':0.25}
WORD_TOKEN = re.compile(r'\b(?:'+ '|'.join(WORDS)+r')\b',re.I)

def numeric_spans(text):
    """Ranges retain their separator; grouped numbers and exponents are one slot."""
    spans=[{'start':m.start(),'end':m.end(),'surface':m.group(),'value':float(m.group().replace(',','')),'kind':'digits'} for m in DIGITS.finditer(text)]
    words=list(WORD_TOKEN.finditer(text));i=0
    while i<len(words):
        group=[words[i]];i+=1
        while i<len(words) and re.fullmatch(r'(?:\s+|\s*-\s*|\s+and\s+)',text[group[-1].end():words[i].start()],re.I):
            # Adjacent independent quantities joined by 'and' stay separate
            # unless the preceding phrase includes a magnitude word.
            between=text[group[-1].end():words[i].start()].strip().lower()
            if between=='and' and not any(w.group().lower() in ['hundred','thousand','million'] for w in group):break
            group.append(words[i]);i+=1
        total=current=0
        for w in group:
            v=WORDS[w.group().lower()]
            if v==100:current=max(1,current)*100
            elif v>=1000:total+=max(1,current)*v;current=0
            else:current+=v
        a,z=group[0].start(),group[-1].end()
        if not any(a<s['end'] and z>s['start'] for s in spans):spans.append({'start':a,'end':z,'surface':text[a:z],'value':float(total+current),'kind':'words'})
    ordered=sorted(spans,key=lambda s:s['start']);merged=[]
    for s in ordered:
        if (merged and s['surface'].lower() in ['half','quarter']
                and re.fullmatch(r'\s+and\s+(?:a\s+)?',text[merged[-1]['end']:s['start']],re.I)):
            previous=merged.pop()
            merged.append({'start':previous['start'],'end':s['end'],
                'surface':text[previous['start']:s['end']],
                'value':previous['value']+s['value'],'kind':'compound_fraction'})
        else:merged.append(s)
    return merged

def has_marker(marker,text):
    marker=marker.strip().lower()
    if not marker:return False
    return bool(re.search((r'(?<!\w)' if marker[0].isalnum() else '')+re.escape(marker)+(r'(?!\w)' if marker[-1].isalnum() else ''),text.lower()))

def public_policy(evidence,policies):
    for policy in policies:
        if policy['id']=='GENERIC_NUMERIC' or any(has_marker(str(m),evidence) for m in policy.get('markers',[])):return dict(policy)
    raise ValueError('missing public generic numeric policy')

def replace_all(case,entity,epsilon,registry,random_seed=None):
    if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('M2 requires finite positive epsilon')
    text=entity['value'];spans=numeric_spans(text)
    if not spans:raise ValueError('no numeric value')
    if not all(math.isfinite(s['value']) for s in spans):raise ValueError('non-finite numeric value')
    skeleton=[];last=0
    for s in spans:skeleton.extend([text[last:s['start']],'<NUM>']);last=s['end']
    skeleton.append(text[last:]);skeleton=''.join(skeleton)
    start=int(entity['start']);end=int(entity['end'])
    outer=case['query'][max(0,start-40):start]+' <METRIC_VALUE> '+case['query'][end:end+40]
    type_hint=str(entity.get('operational_type',entity.get('type',''))).lower()
    policies=registry['policies'];fallback=public_policy(skeleton,policies)
    if fallback['id']=='GENERIC_NUMERIC':fallback=public_policy(type_hint+' '+outer,policies)
    specs=[];date_order=None
    if len(spans)==3 and re.fullmatch(r'\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*',text):date_order=['Y','M','D']
    elif len(spans)==3 and re.fullmatch(r'\s*\d{1,2}[-/]\d{1,2}[-/]\d{4}\s*',text):date_order=['M','D','Y']
    for i,s in enumerate(spans):
        # Strip all numeric magnitudes from policy evidence. Unit immediately
        # after this slot takes precedence, then the fixed entity template.
        following=text[s['end']:spans[i+1]['start'] if i+1<len(spans) else len(text)]
        preceding=text[spans[i-1]['end'] if i else 0:s['start']]
        policy=public_policy(following,policies)
        if policy['id']=='GENERIC_NUMERIC' and any(x in preceding for x in ['$','£','€']):
            policy=dict(next(x for x in policies if x['id']=='CURRENCY_DOLLARS'))
        if policy['id']=='GENERIC_NUMERIC':policy=dict(fallback)
        # A monetary suffix is a public unit multiplier: perturb $250k on
        # the dollar policy expressed in thousands, not as 250 dollars.
        if policy['id']=='CURRENCY_DOLLARS':
            unit=re.match(r'\s*([kKmM])\b',following)
            if unit:
                multiplier=1000.0 if unit.group(1).lower()=='k' else 1000000.0
                policy={**policy,**{key:float(policy[key])/multiplier for key in ['low','high','step','scale']}}
                policy['id']='CURRENCY_'+unit.group(1).upper()
        if date_order:
            component=date_order[i]
            low,high,scale={'Y':(1900,2100,5),'M':(1,12,1),'D':(1,31,3)}[component]
            policy={'id':'DATE_'+component,'low':low,'high':high,'step':1,'scale':scale}
        specs.append(policy)
    rng=random.Random(random_seed) if random_seed is not None else secrets.SystemRandom()
    eps_each=epsilon/len(spans);coordinates=[];samples=[]
    for index,(s,policy) in enumerate(zip(spans,specs)):
        low,high,step,scale=[float(policy[k]) for k in ['low','high','step','scale']]
        if not (step>0 and scale>0 and high>=low):raise ValueError('invalid public grid')
        x=min(high,max(low,s['value']));count=int(round((high-low)/step))+1
        grid=[low+j*step for j in range(count)]
        logs=[-eps_each*abs(x-y)/(2*scale) for y in grid];largest=max(logs)
        chosen=rng.choices(range(count),weights=[math.exp(z-largest) for z in logs],k=1)[0]
        samples.append(grid[chosen]);coordinates.append({'index':index,'start':s['start'],'end':s['end'],'epsilon':eps_each,
            'metric_kind':policy['id'],'scale':scale,'public_bounds':[low,high],'public_step':step,'domain_size':count,
            'sampled_index':chosen,'input_clipped_to_public_domain':x!=s['value']})
    post=[]
    if date_order:
        y=int(samples[date_order.index('Y')]);month=int(samples[date_order.index('M')]);di=date_order.index('D')
        samples[di]=min(samples[di],calendar.monthrange(y,month)[1]);post.append('clamp sampled day to sampled calendar month')
    parts=[];last=0
    for s,policy,y,c in zip(spans,specs,samples,coordinates):
        # Precision derives from the public grid, never the private magnitude.
        precision=max(0,len(f"{float(policy['step']):.10f}".rstrip('0').split('.')[-1]))
        rendered=f'{y:.{precision}f}' if precision else str(int(round(y)))
        parts.extend([text[last:s['start']],rendered]);last=s['end'];c['replacement']=rendered
    parts.append(text[last:]);replacement=''.join(parts)
    return replacement,{'epsilon':epsilon,'mechanism':'coordinate_product_exponential_mLDP','numeric_count':len(spans),
        'epsilon_per_coordinate':eps_each,'epsilon_sum':sum(c['epsilon'] for c in coordinates),
        'metric':'mean_i(abs(x_i-x_i_prime)/scale_i)','coordinates':coordinates,'postprocessing':post,
        'privacy_scope':'conditional on fixed public numeric count, template, units, coordinate policies and grids; not whole-text or router DP',
        'all_numeric_spans_processed':True,'scale':coordinates[0]['scale'],'metric_kind':coordinates[0]['metric_kind'] if len(spans)==1 else 'VECTOR',
        'public_bounds':coordinates[0]['public_bounds'] if len(spans)==1 else [c['public_bounds'] for c in coordinates],
        **({key:coordinates[0][key] for key in ['public_step','domain_size','sampled_index','input_clipped_to_public_domain']}
           if len(spans)==1 else {})}
