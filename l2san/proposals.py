import re, json
from . import client as q

GEN = """Fill only the TARGET marker in a task whose protected spans are masked.
Return one plausible replacement text, preserving the grammatical and semantic
role of that slot. The replacement may be a name, phrase, event, proposition,
clinical finding or clause. It need not belong to any dictionary. Use enough
words to express it faithfully; there is no five-word restriction. Do not answer
the task, explain, output a list, or output a type label or another placeholder.
Output the replacement text only, without quotation marks."""

ROLE = """Check replacements for TARGET using only the supplied public masked
task, target type and immediate left/right context. Keep candidates that can
occupy THIS slot with the same grammatical and semantic role. Reject answers,
advice, repetitions/continuations of the rest of the task, lists of alternatives,
placeholders and fabricated marker strings. Clauses and longer expressions are
allowed when the slot calls for them. Do not guess the hidden original value.
Return JSON only: {"keep":[0,1]}. An empty keep list is allowed."""

SCORE = """Score every supplied replacement against the PRIVATE original target
in its original task. semantic and task are integers from 0 to 100.
semantic measures preservation of the target's actual meaning, specificity,
negation and relations. Identity or a true equivalent gets 100; a truthful
generalization gets 40-80; a related but different fact gets 0-35.
task measures preservation of the correct conclusion/action after local surface
reference restoration. Preserve clues and option identities; never reward turning
a symptom into an inferred diagnosis or a distractor into the correct answer.
Do not score leakage/privacy risk. Output all candidate indices exactly once:
{"scores":[{"id":0,"semantic":90,"task":80}]}. No explanation."""

def clean(text):
    text = ' '.join(text.strip().strip('"`').split())
    if not text or any(s in text for s in ['⟦','⦅','[TARGET]','<think>','</think>']): return None
    return text

def public_logprobs(masked, left, right, candidates, local):
    """Same teacher-forced model conditional for candidates from both sources."""
    prefix = 'Task with protected spans masked:\n'+masked+'\nFill only [TARGET]. Replacement text: '
    output = {}
    for start in range(0,len(candidates),16):
        chunk = candidates[start:start+16]
        raw = q.completion_request({'model':local.local_model,'prompt':[prefix+c for c in chunk],
                                    'max_tokens':0,'echo':True,'logprobs':1,'temperature':0},local)
        choices = {int(r['index']):r for r in raw['choices']}
        suffix = q.suffix_logprobs(left, right, chunk, local)
        for i,c in enumerate(chunk):
            info=choices[i]['logprobs'];scores=[]
            for offset,token,score in zip(info['text_offset'],info['tokens'],info['token_logprobs']):
                if score is not None and int(offset)+len(str(token))>len(prefix):scores.append(float(score))
            if not scores: raise ValueError('No candidate token logprobs returned')
            conditional=sum(scores)/len(scores)
            output[c]={'candidate_logprob':conditional,'right_context_logprob':suffix[c],
                       'public_logprob':.5*conditional+.5*suffix[c], 'candidate_tokens':len(scores)}
    return output
