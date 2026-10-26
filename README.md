# L2San

Entity-aware, task-adaptive sanitization for local–cloud LLM collaboration.

- `l2san/`: routing and M1–M4 mechanisms, plus local recovery.
- `router/`: frozen Laya inference and the small trained adapter.
- `configs/`: numeric policies and public broad candidates.
- `data/router_labeled.jsonl`: 431 labeled queries.
- `data/router_split.json`: train/development/old-test split (274/69/88).
- `data/test200.jsonl`: 200 annotated legal/medical questions, 100 per domain.

## Run

Python 3.10+ and local Qwen3.5-4B services are required.

```bash
pip install -r requirements.txt
hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --local-dir models/laya
python -m l2san router-input --input data/test200.jsonl --output runs/input.jsonl
python router/predict.py --base-model models/laya --device cuda:0 \
  --input runs/input.jsonl --output runs/routes.jsonl
python -m l2san sanitize --input data/test200.jsonl \
  --predictions runs/routes.jsonl --s 0.5 --output runs/s05
```

Use `python -m l2san --help` to configure local endpoints. M3 requires chat and
completion logprobs; M4 requires vLLM structured outputs and thinking controls.
Only `public/` outputs go to the cloud; keep `private/` outputs local.

## Implementation

M1 uses untyped placeholders; M2 perturbs all numeric coordinates; M3 generates
masked-context candidates and samples with tie-averaged winner randomized response;
M4 mixes same-task requests with local fidelity checks. Equivalent mentions share
a replacement. The policy is the old routing configuration with optimized M4.

M3 expands a candidate pool toward 192, retains up to 96, then uses a fixed public
subset of 32. It is not direct32 generation. M4 uses k=2 below s=.25 and k=4 otherwise;
peer generation enables thinking, normalization disables it. Supported low-s anchors
are 0, .25, .45, .5, .525, .55, .625 and .74; [.75,1] uses the original policy.
At s=1, all entities use M3 with 32 public broad candidates and epsilon=0.

DP claims are conditional on fixed public support/templates; they do not cover
routing or imply entity DP for M4. This portable refactor does not distribute
historical releases or guarantee identical sampled outputs.

The 200-query cohort was selected for local-Qwen failure/cloud-Terra success and
subsequently used in exploratory development. It is not untouched validation.
It has no exact ID or normalized-text overlap with the 431-query corpus.
References are evaluation-only and are not sanitizer inputs. Source question text
and upstream models retain their respective rights; the MIT code license does not
relicense them. No experiment results, evaluation scripts or credentials are included.
