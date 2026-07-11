# Event-Driven Geometric Continual Learning

`chaos.continual` is the transactional successor to the stage-specific
`qwen35_lifelong_pipeline.py` experiment. It keeps that script intact as a
regression fixture while moving orchestration into model- and task-agnostic
interfaces.

## Invariants

1. The committed model is read-only while an event is being acquired.
2. Reward-only trajectories never contain a gold target field.
3. Successes, failures, invalid completions, and errored rollouts remain in the audit ledger.
4. Every rollout group is pinned to one temporary-policy version.
5. Historical training IDs cannot enter a new event update.
6. Explicit revisions release only declared profiles plus their dependency closure.
7. Overlap alone can report a conflict but cannot erase a protected capability.
8. A candidate becomes current only after capability, retention, general, numerical, access, budget, and revision-staleness gates pass.
9. `current.json` is published last. A crash before publication leaves the previous model authoritative.

## Event Lifecycle

```text
queued -> routed -> acquiring -> profiling -> consolidating -> evaluating
       -> committed | rejected | blocked
```

An event may provide demonstrations, a verifier, or both. Revision intent is
orthogonal to the learning signal: a changed policy may be acquired from a
demonstration, reward, or a hybrid of the two.

```python
LearningEvent(
    event_id="preference_revision",
    revision=2,
    kind="revision",
    examples=dataset,
    targets=TargetRef(visibility="verifier_only"),
    verifier=VerifierSpec("revision_exact"),
    dependencies=("capability:tooluse:r0",),
    supersedes=("capability:preference:r1",),
)
```

Targets with `verifier_only` visibility are available only inside the verifier.
The rollout engine receives `RewardResult`, not the target or parser.

## Runtime Layout

The local Hugging Face runtime is deliberately synchronous for mechanism proof:

```text
CPU       event orchestration, verifiers, manifests, transaction store
cuda:0    temporary adapter or full candidate student
cuda:1    frozen committed anchor and acquired temporary teacher
```

It uses float32 training, `AdamW(foreach=False)`, completion-only learning
masks, grouped on-policy credit, adaptive anchor KL, rolling acquisition and
consolidation checkpoints, exact generated token IDs, and CPU safetensor
geometry profiles. Adapter pressure is compared directly with stored profile
bases to measure overlap, occupied union rank, and post-projection residual
energy. This borrows
Prime-RL's typed rollout, policy version, algorithm, and process-boundary ideas
without enabling asynchronous policy lag or its distributed vLLM/FSDP stack.

## Artifacts

```text
STORE/
  current.json
  data_access.jsonl
  journal/events.jsonl
  profile_blobs/*.safetensors
  canaries/*.json
  transactions/<event>/<attempt>/
  versions/v000000/
  versions/v000001/

EVENTS/
  inbox/
  leased/
  committed/
  rejected/
```

Only the current pointer determines the serving model. Rejected transaction
directories keep their ledgers and diagnostics but cannot alter the pointer or
registry. Full candidate, resume, and temporary-policy weights are pruned after
a commit or rejection; published versions retain the current checkpoint and
recent history without duplicating transaction-local models.

## General Canary

Every production commit requires a frozen general-capability canary. Use
evaluation-only examples that never appear in updates:

```json
[
  {"prompt": "Question: ...\nAnswer:", "target": "..."},
  {"prompt": "Instruction: ...\nResponse:", "target": "..."}
]
```

Use 32-64 stable language, instruction, and reasoning examples for event-time
gating. The larger lm-eval panel remains a periodic/final audit rather than an
optimizer input.

## Kaggle Setup

```bash
export ROOT=/kaggle/working/gpsp_cl_stream
export SCRATCH=/kaggle/temp/gpsp_cl_stream
export STORE="$ROOT/store"
export EVENTS="$ROOT/events"
export EVENT_FILES="$ROOT/event_files"
export MAIN_FINAL=/kaggle/working/gpsp_cl_clean/main_cl/full/qwen35_five_skill_seed1337/amoeba/checkpoints/final
export HISTORY=/kaggle/working/gpsp_cl_clean/manifests/qwen35_shen3_manifest_seed1337.json
export STAGE1_MANIFEST=/kaggle/working/gpsp_cl_clean/lifelong_stage1/manifest.json

mkdir -p "$ROOT" "$SCRATCH" "$EVENT_FILES/history" "$EVENT_FILES/stage1"
export HF_HOME="$SCRATCH/hf"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TMPDIR="$SCRATCH/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m pip install -U 'safetensors>=0.8.0' 'tokenizers==0.22.2'
python -m pip install -U --no-deps git+https://github.com/huggingface/transformers.git
python -m pip install -e '/kaggle/working/chaos[rl,viz]'

python - <<'PY'
import safetensors, tokenizers, transformers
from transformers import AutoConfig

checkpoint = "/kaggle/working/gpsp_cl_clean/main_cl/full/qwen35_five_skill_seed1337/amoeba/checkpoints/final"
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
print("tokenizers", tokenizers.__version__)
print("model_type", config.model_type)
PY
```

Create the frozen `general_canary.json` before training. It must be a non-empty
JSON list of 32-64 held-out records such as
`{"prompt": "...", "target": "...", "family": "instruction"}`. These rows
are evaluation/profile data only and never enter an update. Freeze them as the
first protected geometry event:

```bash
chaos-continual general-canary-event \
  --canary "$ROOT/general_canary.json" \
  --output "$EVENT_FILES/base_general.json"
```

Generate profile-only events for the checkpoint's existing real skills:

```bash
chaos-continual history-events \
  --manifest "$HISTORY" \
  --output-dir "$EVENT_FILES/history"
```

Generate the synthetic acquisition/composition/revision stream:

```bash
chaos-continual stage1-events \
  --manifest "$STAGE1_MANIFEST" \
  --output-dir "$EVENT_FILES/stage1" \
  --acquisition-steps 120 \
  --composition-steps 120 \
  --group-size 4
```

Initialize once from the existing Amoeba checkpoint:

```bash
chaos-continual init \
  --runtime hf \
  --store "$STORE" \
  --model "$MAIN_FINAL" \
  --device cuda:0 \
  --teacher-device cuda:1 \
  --dtype float32
```

Submit the base-general event and then history events. Submission order is
persisted independently of event filenames:

```bash
chaos-continual submit --events "$EVENTS" --event "$EVENT_FILES/base_general.json"

for EVENT in "$EVENT_FILES"/history/*.json; do
  chaos-continual submit --events "$EVENTS" --event "$EVENT"
done
```

Profile and commit existing skills without modifying model weights:

```bash
chaos-continual run \
  --runtime hf \
  --store "$STORE" \
  --events "$EVENTS" \
  --device cuda:0 \
  --teacher-device cuda:1 \
  --dtype float32 \
  --general-canary "$ROOT/general_canary.json" \
  --max-seq-len 160 \
  --bootstrap-profile-samples 16 \
  --min-layers 8
```

Then submit and run the learning events:

```bash
for EVENT in "$EVENT_FILES"/stage1/*.json; do
  chaos-continual submit --events "$EVENTS" --event "$EVENT"
done

chaos-continual run \
  --runtime hf \
  --store "$STORE" \
  --events "$EVENTS" \
  --device cuda:0 \
  --teacher-device cuda:1 \
  --dtype float32 \
  --general-canary "$ROOT/general_canary.json" \
  --target-suffixes mlp.down_proj,mlp.up_proj \
  --adapter-rank 16 \
  --adapter-alpha 32 \
  --acquisition-lr 2e-5 \
  --consolidation-lr 2e-6 \
  --consolidation-steps 120 \
  --consolidation-save-interval 30 \
  --group-size 4 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 10 \
  --max-seq-len 160 \
  --min-layers 8 \
  2>&1 | tee -a "$ROOT/continual_stream.log"
```

The same `run` command resumes an interrupted adapter or consolidation phase.
Committed and already-completed events are idempotently skipped.

## Inspection

```bash
chaos-continual status --store "$STORE" --events "$EVENTS"
chaos-continual audit --store "$STORE"
chaos-continual evaluate --store "$STORE" --version -1
chaos-continual plot --store "$STORE" --output "$ROOT/commit_history.png"
```

## Validation Status

The event contracts, verifier isolation, group-relative credit, dependency
release, measured adapter/profile overlap, atomic commit/rollback, orphan
recovery, queue ordering, exact acquisition resume, profile-only bootstrap,
historical-row rejection, CLI flow, and demonstration-to-revision reference run
are covered by local tests. The Qwen/Hugging Face backend is code-complete but
still requires the real two-T4 run before its learning or retention results can
be claimed.
