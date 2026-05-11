# Experiment Notes

This file is the short reader path through the research scripts.

## 1. Alien Ladder Stress Test

Script:

```text
alien_ladder_cl_audit.py
```

Setup:

```text
Model: Qwen/Qwen2.5-0.5B
Task ladder: SCAN -> COGS -> GeoQuery
Baselines: naive SFT, SDFT-style baseline
Constraint: old_task_examples = 0, proxy_batches = 0
```

Main result:

```text
Naive SFT and SDFT learned the new GeoQuery task, but damaged old SCAN/COGS behavior.
Amoeba no-proxy retained substantially more old-task behavior while keeping GeoQuery competitive.
```

Safe public wording:

```text
In the Alien Ladder stress test, Amoeba controlled catastrophic forgetting of measured old skills under zero old-task replay and zero proxy corpus.
```

## 2. Qwen No-Proxy Desiderata Audit

Script:

```text
qwen_cl_desiderata_audit.py
```

Setup:

```text
base A -> train B adapter teacher -> consolidate B into base_AB
base_AB -> train D adapter teacher -> consolidate D with no old examples / no proxy
```

Key result:

```text
B-field survived D consolidation: 1.000 -> 0.969
old_task_examples = 0
proxy_batches = 0
PPL ratio stayed near 1.028 in the focused run
```

Interpretation:

```text
This is the core real-model replay-free consolidation result.
```

## 3. Qwen Lateral Skill Propagation

Script:

```text
qwen_cl_desiderata_audit.py
```

Observation:

```text
Preserving two skills does not automatically create a direct composed circuit.
```

Result:

```text
LSP generated component traces, parsed/verified them, and distilled the scaffold into direct composition.
Composition final-token reached 0.306 where direct composition was previously weak.
```

Safe public wording:

```text
Composition was not magic zero-shot. It required scaffold-to-weight transfer after the component skills were individually reliable.
```

## 4. Gemma Cross-Family Transfer

Script:

```text
gemma_cl_desiderata_audit.py
```

Setup:

```text
Model: google/gemma-3-270m
The original Qwen-style interface failed.
The task surface was recalibrated for Gemma.
```

Key result:

```text
base_AB B-field: 0.844
D no-proxy B-field: 0.729
D no-proxy sort token: 0.361
D no-proxy PPL ratio: 1.088
old_task_examples = 0
proxy_batches = 0
```

Safe public wording:

```text
The replay-free mechanism is not Qwen-specific; it transferred to Gemma after interface calibration and acquisition gating.
```

## 5. Qwen Expansion / Early Pipeline

Script:

```text
qwen_continual_proof.py
```

Purpose:

```text
Earlier staged Qwen pipeline exploring B/D skills, consolidation, composition variants, and expansion rescue.
```

Key signal:

```text
Expansion beat the fixed update on the sort/B-retention tradeoff in the canonical expansion run.
```

Safe public wording:

```text
Expansion showed the first pretrained-model signal that capacity growth can rescue learning when fixed geometry is too crowded.
```

## 6. GSM8K -> Sort Baseline Comparison

Script:

```text
gsm8k_sdft_baseline_audit.py
```

Setup:

```text
Model: Qwen/Qwen2.5-0.5B
Old task: GSM8K subset
New task: proof_v2 stable sort
Baselines: naive SFT and SDFT-style self-distillation
```

Key result:

```text
GSM8K exact stayed 0.125 -> 0.125 under Amoeba no-proxy.
Amoeba had the best new sort acquisition: 0.172 vs SFT 0.098 vs SDFT 0.006.
```

Safe public wording:

```text
This is a retention/acquisition tradeoff test on a recognizable task, not a claim that Qwen 0.5B is a strong GSM8K solver.
```

## 7. Qwen Z-Law / Z-Guided Expansion

Script:

```text
qwen_z_law_controller_audit.py
```

Setup:

```text
Old skill: proof_v2 record routing
New skill: tagged conflict task from the same records
```

Key result:

```text
Fixed no-growth learned the new task but forgot the old skill:
old = 0.000, new = 1.000

Z-selected frozen-base expansion preserved old skill and learned new:
old = 0.751, new = 1.000
```

Safe public wording:

```text
This is the cleanest Qwen evidence that Z-guided adaptive plasticity can choose growth when fixed geometry cannot safely hold both behaviors.
```

## Method Summary

```text
Amoeba protects old skills.
Z-Tomography routes plasticity.
LSP makes learned skills compose.
Expansion grows capacity when the fixed model is crowded.
```

That is Eigenesis.
