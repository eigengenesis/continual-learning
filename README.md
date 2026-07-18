# Continual Learning

Research code for replay-free, in-weights continual learning in small language
models. The project studies whether new capabilities can be acquired and
consolidated without replaying historical task data or catastrophically
overwriting previously learned behavior.

The current system combines temporary skill acquisition, layerwise geometric
profiling, occupied-subspace projection, old-checkpoint anchoring, selective
profile release, and transactional model commits.

## Research Status

This repository contains two complementary implementations:

1. A stage-aware experimental runner used to develop and audit the mechanism.
2. A task-agnostic, event-driven continual-learning engine intended to turn the
   mechanism into a coherent long-running system.

The codebase has local unit and integration coverage. Reported Qwen and Gemma
results are scoped to the configurations documented in [EXPERIMENTS.md](EXPERIMENTS.md).
The event-driven Hugging Face backend is implemented and locally contract-tested;
its complete two-T4 scientific run remains pending. This repository does not
claim that continual learning is solved generally.

## Core Method

For each learning event, the system:

1. Acquires the new behavior in a temporary adapter policy from demonstrations,
   verifier rewards, or both.
2. Measures update pressure, overlap with protected profiles, occupied rank,
   and residual update energy across model layers.
3. Produces an immutable geometry plan that protects active capabilities and,
   for explicit revisions, releases only superseded profiles plus their
   dependency closure.
4. Consolidates the temporary policy into a candidate model using new-policy
   alignment, old-checkpoint anchoring, and occupied-gradient projection.
5. Commits the candidate atomically only after capability, retention, general,
   staleness, numerical, access, and budget gates pass.

Acquisition never mutates the committed model. Rejected events retain their
audit records but cannot advance the serving checkpoint.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/chaos/continual/` | Event contracts, verifier plugins, grouped on-policy acquisition, geometry control, consolidation, durable streams, and atomic commit/rollback. |
| `qwen35_lifelong_pipeline.py` | Resumable Stage-1 reference lifecycle: primitive acquisition, composition, policy revision, selective release, and final audit. |
| `qwen35_five_skill_cl_audit.py` | Frozen-manifest SFT, SDFT, OP-SDFT, and Amoeba comparison runner. |
| `alien_ladder_cl_audit.py` | SCAN -> COGS -> GeoQuery stress test and shared consolidation utilities. |
| `qwen_tomography.py` | Layerwise activation/gradient SVD profiles, pressure scoring, and saturation reports. |
| `standalone_latent_lora_qwen.py` | Shared latent-adapter implementation and model utilities. |
| `src/chaos/` | Installable thermodynamic training-control library used by the broader Eigenesis work. |
| `docs/continual_pipeline.md` | Event-engine architecture, invariants, artifact layout, and exact two-T4 runbook. |
| `tests/` | Unit, privacy, interruption, recovery, access-control, geometry, and end-to-end transaction tests. |

Earlier Qwen/Gemma audits remain in the repository as research history and
cross-checks. They are not the primary orchestration interface.

## Installation

For local development:

```bash
python -m pip install -e ".[dev]"
```

For the Hugging Face continual-learning runtime:

```bash
python -m pip install -e ".[dev,rl,viz]"
```

Qwen 3.5 may require a recent Transformers build. The validated Kaggle setup and
dependency preflight are documented in [docs/continual_pipeline.md](docs/continual_pipeline.md).

## Validation

Run the local suite:

```bash
python -m pytest -q
python -m py_compile src/chaos/continual/*.py
chaos-continual --help
```

The suite covers, among other invariants:

- verifier-private targets do not enter optimizer-visible metadata;
- grouped rollouts retain successes, failures, invalid outputs, and errors;
- generated token IDs are not silently changed by decode/re-tokenize cycles;
- historical committed training rows cannot be replayed into later updates;
- explicit revision release follows profile dependency closure;
- interrupted acquisition resumes from matching policy, optimizer, RNG, and
  ledger state;
- a failed event leaves the current model pointer unchanged;
- orphaned versions and pointer-published transactions recover safely.

## Event-Driven Engine

The CLI exposes a durable event queue and a single evolving model store:

```bash
chaos-continual init --runtime hf --store "$STORE" --model "$MODEL"
chaos-continual submit --events "$EVENTS" --event event.json
chaos-continual run --runtime hf --store "$STORE" --events "$EVENTS" \
  --device cuda:0 --teacher-device cuda:1 --dtype float32 \
  --general-canary general_canary.json
chaos-continual status --store "$STORE" --events "$EVENTS"
chaos-continual audit --store "$STORE"
```

Frozen Stage-1 and historical-skill manifests can be converted into generic
events with `stage1-events` and `history-events`. See the full
[continual pipeline runbook](docs/continual_pipeline.md) before starting a GPU
run; it includes the required base-general profile, dependency ordering,
checkpoint layout, and resume procedure.

## Stage-1 Reference

The legacy runner remains available as a regression fixture and mechanism
audit. It freezes its manifest and hyperparameters before training:

```bash
python -u qwen35_lifelong_pipeline.py \
  --mode build_manifest \
  --model-id "$MODEL" \
  --history-manifest "$HISTORY" \
  --manifest-path "$PIPELINE/manifest.json"

python -u qwen35_lifelong_pipeline.py \
  --mode run \
  --model-id "$MODEL" \
  --history-manifest "$HISTORY" \
  --manifest-path "$PIPELINE/manifest.json" \
  --output-dir "$PIPELINE" \
  --device cuda:0 \
  --teacher-device cuda:1 \
  --dtype float32 \
  --resume auto
```

Use `--stop-after <stage>` to divide the lifecycle across bounded notebook
sessions. Starting the same command with `--resume auto` verifies committed
stage hashes and restores an interrupted stage from its latest valid snapshot.

## Reproducibility Principles

- Manifests, dataset rows, seeds, verifier configurations, and gates are
  fingerprinted before training.
- Reward-only targets remain verifier-private and gold rollout rescue is
  disabled.
- Access logs record every row used for an update.
- General and historical canaries are evaluation/profile data, never replay
  batches.
- `current.json` is the publication boundary for the model store and is updated
  only after candidate artifacts and checksums are complete.
- Changed configurations are new event revisions; failed gates are not silently
  weakened and retried.

## Background

The initial public writeup predates the transactional engine and records the
development path that led to this repository:
[Replay-Free Continual Learning](https://x.com/eigengenesis/status/2053855070551437495).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

Until a formal paper citation is available, cite the repository:

```text
Eigenesis Continual Learning
https://github.com/eigengenesis/continual-learning
```
