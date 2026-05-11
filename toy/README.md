# Toy And Mechanism Experiments

These are the prototype experiments behind the Eigenesis continual-learning pipeline.

They are not the headline Qwen/Gemma benchmark scripts. They are mechanism tests: early Z guards, Water Weights, replay independence, no-proxy consolidation, expansion, LSP/composition, Z-law controllers, tomography falsification, alien-language composition, verifier-based self-acquisition, grokking, and internal shape-engine probes.

## Why This Folder Exists

The root repo contains the main pretrained-model scripts.

This folder contains the toy ladder that produced the mechanisms:

```text
Z pressure signal
-> gradient / latent geometry
-> adapter consolidation
-> replay independence
-> no-proxy Amoeba
-> lateral skill propagation
-> expansion gating
-> Z-guided growth
-> alien composition / verifier tests
```

## Reader Path

If you only read a few:

| file | why it matters |
| --- | --- |
| `colab_replay_independence_toy.py` | Shows replay itself was not the carrier of retention; preservation signal during consolidation was the important part. |
| `toy_cuda_old_example_free_table.csv` | Locked toy table for no-old-example / no-proxy coexistence using old-checkpoint KL + hidden anchors. |
| `toy_auto_expansion_controller_lab.py` | Clean automatic expansion proof: fixed model forgets old skills, expansion preserves old and learns new. |
| `z_law_toy_audit.py` | Corrected Z-law toy audit: Z acts as a task-conditioned pressure sensor for branch/growth decisions. |
| `toy_alien_language_cl_probe.py` | Shows scaffolded B -> D composition can be distilled into direct C behavior. |
| `toy_verifier_self_acquisition_lab.py` | Shows naive self-distillation fails for absent skills; verifier + expansion can acquire new behavior. |

## Full Map

| file | mechanism |
| --- | --- |
| `colab_z_forgetting_guard_benchmark.py` | Early Z forgetting guard / replay trigger. |
| `colab_water_weights_benchmark.py` | Early Water Weights stack: viscosity, old-gradient anchors, adapters, bounded replay. |
| `colab_water_weights_lateral_consolidation_benchmark.py` | Adapter skill storage and dual-teacher lateral consolidation. |
| `colab_phase_reversibility_lab.py` | Phase separation, consolidation, extraction, and partial reversibility. |
| `colab_layer_expansion_lateral_propagation_lab.py` | Early live expansion block / lateral propagation tests. |
| `colab_layer_expansion_lateral_lab.py` | First clean toy expansion success. |
| `colab_compositional_transfer_lab.py` | A -> B -> C -> D transfer and fixed-capacity bottleneck. |
| `colab_compositional_expansion_lab.py` | Expansion rescue for compositional bottlenecks. |
| `colab_replay_independence_toy.py` | Replay-dependence ablation. |
| `toy_cuda_old_example_free_table.csv` | No-proxy toy result table. |
| `colab_skill_affinity_toy.py` | Skill-family locality / affinity. |
| `colab_skill_anchor_affinity_toy.py` | Related-anchor learning effects. |
| `colab_skill_universality_toy.py` | Cross-seed related-anchor universality probe. |
| `toy_interface_composition_probe.py` | Typed interfaces for scaffolded composition. |
| `toy_meta_automaticity_probe.py` | Composition automaticity vs scaffold-to-weight transfer. |
| `toy_auto_expansion_controller_lab.py` | Automatic expansion governor. |
| `z_law_toy_audit.py` | Z-guided expansion controller. |
| `z_tomography_falsification_audit.py` | Falsification test for naive "top-Z is best" claim. |
| `z_tomography_occupancy_audit.py` | Occupied-geometry protection test. |
| `toy_alien_language_cl_probe.py` | Alien-language LSP / scaffolded composition. |
| `toy_verifier_self_acquisition_lab.py` | Verifier-grounded self-acquisition. |
| `accelerated_grokking_audit.py` | Structure-aware gradient shaping for grokking. |
| `internal_shape_engine_audit.py` | Learned neural shape engine probe. |

## Claim Boundary

These scripts are exploratory research artifacts. They are useful because they isolate mechanisms under controlled conditions.

The pretrained-model evidence lives in the root scripts:

```text
qwen_cl_desiderata_audit.py
gemma_cl_desiderata_audit.py
alien_ladder_cl_audit.py
```

