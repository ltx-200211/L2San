import re
from . import client, m2

SUBTYPE_PROMPT = """Choose the implementation subtype for sensitive entities ALREADY assigned to M23.
Never decide M1/M4 and never use the original router's C2/C3 labels.
M2 perturbs ALL explicit numerical magnitudes in an entity, retaining its nonnumeric
text template. Use M2 only when the sensitive fact is genuinely numerical (age,
duration, amount, measurement, date, count), and altering all parsed numeric values
is a meaningful protection for this entity. A disease code/name containing digits,
an address/identifier, or a whole event with incidental numbers is normally M3.
M3 replaces the complete entity with a semantic text candidate. Use M3 for
qualitative facts, names, relationships, diagnoses and other text.
For M2, quantities must be recoverable from supplied parsed numeric spans; do not
invent an implicit value or a new normalization. A/an temporal units are normalized
to 1 by a fixed preprocessing rule. Context is provided only for disambiguation.
Return JSON only: {"entities":[{"id":"e1","route":"M2","reason":"short"}]}.
Return exactly every supplied entity once. No task answer."""

def numeric_text(e):
    return re.sub(r'\b(?:a|an)\s+(?=(?:year|month|week|day|hour|minute|second)\b)',
                  '1 ',e['value'],flags=re.I)

def classify(case, local):
    units = case['units']
    payload = {'query': case['query'], 'entities': [dict(id=e['id'], value=e['value'],
        type=e['type'], numeric_spans=m2.numeric_spans(numeric_text(e))) for e in units]}
    raw = client.local_json(SUBTYPE_PROMPT, payload, local, max(1000,110*len(units)))['entities']
    by = {e['id']:e['route'] for e in raw}
    if len(raw)!=len(units) or set(by)!={e['id'] for e in units} or any(v not in ('M2','M3') for v in by.values()):
        raise ValueError('Incomplete subtype prediction')
    return {e['id']: ('M3' if by[e['id']]=='M2' and not m2.numeric_spans(numeric_text(e)) else by[e['id']]) for e in units}
