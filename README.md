# Eigenesis Continual Learning

Research scripts for replay-free continual learning experiments on small pretrained language models.

This repo is a research artifact, not a polished library. The scripts are intentionally close to the experiments: they train adapter teachers, collect layerwise tomography, project gradients away from occupied subspaces, consolidate skills with old-checkpoint anchors, and benchmark against naive SFT / SDFT-style baselines.

## Article

Public writeup:

https://x.com/eigengenesis/status/2053855070551437495

## What This Tests

The core question:

> Can a model learn a new skill without replaying old task data and without catastrophically destroying the old skill?

The current answer is scoped but real:

- Qwen 0.5B no-proxy continual learning works on staged synthetic skills.
- Gemma 270M shows the same replay-free pattern after interface calibration.
- Alien Ladder stress tests SCAN -> COGS -> GeoQuery against SFT and SDFT.
- Z-Tomography measures layer pressure and occupied/free geometry.
- Amoeba consolidation uses same-batch old-checkpoint anchoring plus gradient projection.

## Repository Map

| file | purpose |
| --- | --- |
| `alien_ladder_cl_audit.py` | Main stress test: SCAN -> COGS -> GeoQuery, comparing naive SFT, SDFT baseline, fixed no-proxy, and expanded no-proxy. |
| `qwen_cl_desiderata_audit.py` | Focused Qwen no-proxy continual-learning audit: adapter acquisition, Amoeba consolidation, projection, LSP composition branches. |
| `qwen_continual_proof.py` | Earlier large Qwen pipeline with staged synthetic skills, consolidation, composition, and expansion experiments. |
| `gemma_cl_desiderata_audit.py` | Cross-family Gemma 270M no-proxy audit with Gemma-calibrated task interface. |
| `gsm8k_sdft_baseline_audit.py` | GSM8K -> Sort retention/acquisition comparison against naive SFT and SDFT-style baseline. |
| `qwen_z_law_controller_audit.py` | Qwen Z-law / Z-guided expansion audit for conflict pressure and adaptive capacity growth. |
| `qwen_tomography.py` | Layerwise activation/gradient SVD profiles, occupied bases, pressure scoring, layer selection, saturation reports. |
| `standalone_latent_lora_qwen.py` | Shared model loading, latent adapter, LoRA-style modules, schedules, and utility code. |
| `EXPERIMENTS.md` | Human-readable experiment summaries and safest public claims. |

## Quick Sanity Check

This only checks that the scripts import and compile:

```bash
python -m py_compile \
  standalone_latent_lora_qwen.py \
  qwen_tomography.py \
  qwen_continual_proof.py \
  qwen_cl_desiderata_audit.py \
  gemma_cl_desiderata_audit.py \
  gsm8k_sdft_baseline_audit.py \
  qwen_z_law_controller_audit.py \
  alien_ladder_cl_audit.py
```

## Install

Use a GPU environment. The largest experiments were run with CUDA and `bfloat16`.

```bash
python -m pip install -r requirements.txt
```

Optional but recommended:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Main Run

The flagship script is:

```bash
python -u alien_ladder_cl_audit.py \
  --device cuda \
  --dtype bfloat16 \
  --seed 1337
```

The exact long-form commands used during research changed across runs. See `EXPERIMENTS.md` for the result summaries and interpretation.

## Core Ideas

- **Z-Tomography**: scan layers using activation and gradient geometry to estimate pressure, occupancy, and free capacity.
- **Occupied-manifold projection**: remove update components that point through old-skill geometry.
- **Amoeba consolidation**: consolidate a new teacher into the base model using only the new-task batch while an old checkpoint anchors behavior and hidden geometry.
- **Lateral Skill Propagation**: compose already-learned skills through scaffolded traces, then distill the composition into one forward pass.
- **Expansion gating**: grow capacity when fixed model geometry is crowded.

## Scope

This repo does not claim that continual learning is solved universally.

It provides evidence that replay-free skill retention is possible in these tested Qwen/Gemma settings, and that gradient geometry can be used as a control surface for continual learning.

## Citation / Contact

If you use these scripts or ideas, please cite/link the repo:

```text
Eigenesis Continual Learning
https://github.com/eigengenesis/continual-learning
```
