#!/usr/bin/env python3
"""Z tomography law-candidate toy audit.

This is not testing the cartoon claim:

    "top-Z layer always learns fastest."

It tests the claim actually used by the Amoeba / Water Weights controller:

    Z is a task-conditioned pressure sensor. In a constrained learner it should
    stably identify where old/new learning pressure accumulates, and a branch
    governor should be able to use that signal to choose expansion that improves
    the old-skill/new-skill Pareto tradeoff.

The audit reuses the already-proven toy transformer setting:

    A = copy sequence
    B = reverse sequence
    C = sort sequence

The base model first learns A+B. A fixed-size C learner typically acquires C and
forgets A+B. The Z-guided controller measures old/new gradient pressure, tests
growth candidates, and accepts a lateral expansion only when it verifies a better
Pareto branch.

Artifacts:
  z_law_toy_results.csv
  z_law_toy_curves.csv
  z_law_toy_events.json
  z_law_toy_pressure.csv
  z_law_toy_verdict.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import toy_auto_expansion_controller_lab as toy


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(seed_text: str) -> List[int]:
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def candidate_name_for_layer(layer: int, count: int = 1) -> str:
    return f"expand_l{int(layer)}_x{int(count)}"


def pressure_stability_probe(
    model: toy.TinyLM,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Measure whether the pressure top-layer is stable across probe batches."""
    probe_rows: List[Dict[str, object]] = []
    top_layers: List[int] = []
    layer_values: Dict[str, List[float]] = {}

    for probe_idx in range(int(args.pressure_probes)):
        where, top_pressure, pressures = toy.grad_pressure(
            model,
            args,
            seed + 50_000 + probe_idx * 997,
        )
        top_layers.append(int(where))
        for layer, value in pressures.items():
            layer_values.setdefault(str(layer), []).append(float(value))
        sorted_layers = sorted(pressures.items(), key=lambda item: float(item[1]), reverse=True)
        second_pressure = float(sorted_layers[1][1]) if len(sorted_layers) > 1 else 0.0
        gap = float(top_pressure - second_pressure)
        gap_ratio = gap / max(float(top_pressure), 1e-9)
        probe_rows.append(
            {
                "seed": seed,
                "probe": probe_idx,
                "top_layer": int(where),
                "top_pressure": float(top_pressure),
                "second_pressure": second_pressure,
                "gap": gap,
                "gap_ratio": gap_ratio,
                **{f"pressure_l{layer}": float(value) for layer, value in pressures.items()},
            }
        )

    counts = Counter(top_layers)
    mode_layer, mode_count = counts.most_common(1)[0]
    mean_by_layer = {layer: float(np.mean(values)) for layer, values in layer_values.items()}
    mean_sorted = sorted(mean_by_layer.items(), key=lambda item: float(item[1]), reverse=True)
    mean_top_layer = int(mean_sorted[0][0])
    mean_top_pressure = float(mean_sorted[0][1])
    mean_second_pressure = float(mean_sorted[1][1]) if len(mean_sorted) > 1 else 0.0
    mean_gap_ratio = (mean_top_pressure - mean_second_pressure) / max(mean_top_pressure, 1e-9)

    summary: Dict[str, object] = {
        "seed": seed,
        "pressure_top_mode_layer": int(mode_layer),
        "pressure_top_mode_fraction": float(mode_count / max(len(top_layers), 1)),
        "pressure_mean_top_layer": int(mean_top_layer),
        "pressure_mean_top": mean_top_pressure,
        "pressure_mean_second": mean_second_pressure,
        "pressure_mean_gap_ratio": float(mean_gap_ratio),
        "pressure_top_sequence": ",".join(str(item) for item in top_layers),
        **{f"mean_pressure_l{layer}": value for layer, value in mean_by_layer.items()},
    }
    return summary, probe_rows


def extract_first_event(event_rows: Sequence[Dict[str, object]]) -> Dict[str, Any] | None:
    if not event_rows:
        return None
    # There should be only one expansion event by design; keep the helper robust.
    return dict(sorted(event_rows, key=lambda row: int(row.get("step", 0)))[0])


def candidate_lookup(event: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    candidates = event.get("candidates", [])
    if not isinstance(candidates, list):
        return {}
    return {str(item.get("name")): dict(item) for item in candidates if isinstance(item, dict)}


def score_gap(candidates: Dict[str, Dict[str, Any]], winner: str, loser: str) -> float:
    if winner not in candidates or loser not in candidates:
        return float("nan")
    return float(candidates[winner].get("score", 0.0)) - float(candidates[loser].get("score", 0.0))


def best_growth_for_layer(candidates: Dict[str, Dict[str, Any]], layer: int) -> Dict[str, Any] | None:
    prefix = f"expand_l{int(layer)}_x"
    matches = [item for name, item in candidates.items() if name.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda item: float(item.get("score", -1e9)))


def run_seed(args: argparse.Namespace, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    print("=" * 96)
    print(f"Z LAW TOY AUDIT seed={seed}")
    print("=" * 96)
    toy.set_seed(seed)

    curves: List[Dict[str, object]] = []
    old = toy.train_old_model(args, seed)
    old_metrics = toy.evaluate_all(old, args, seed + 333)
    print(
        f"[old_final] A={old_metrics.A_exact:.3f} B={old_metrics.B_exact:.3f} "
        f"C={old_metrics.C_exact:.3f} old={old_metrics.old_mean:.3f}",
        flush=True,
    )

    pressure_summary, pressure_rows = pressure_stability_probe(old, args, seed)
    print(
        f"[z_pressure] top_mode_layer={pressure_summary['pressure_top_mode_layer']} "
        f"mode_fraction={pressure_summary['pressure_top_mode_fraction']:.3f} "
        f"mean_top_layer={pressure_summary['pressure_mean_top_layer']} "
        f"gap_ratio={pressure_summary['pressure_mean_gap_ratio']:.3f}",
        flush=True,
    )

    fixed = toy.train_fixed_branch(old, args, seed, curves)
    auto, raw_events = toy.train_auto_branch(old, args, seed, curves)
    event_rows = [{"seed": seed, **asdict(event)} for event in raw_events]
    first_event = extract_first_event(event_rows)

    selected_candidate = "none"
    selected_layer = -1
    selected_score_gap_vs_no_growth = float("nan")
    top_pressure_layer = int(pressure_summary["pressure_mean_top_layer"])
    top_growth_name = candidate_name_for_layer(top_pressure_layer, 1)
    top_growth_score_gap_vs_no_growth = float("nan")
    top_growth_old = float("nan")
    top_growth_c = float("nan")
    no_growth_old = float("nan")
    no_growth_c = float("nan")

    if first_event is not None:
        candidates = candidate_lookup(first_event)
        selected_candidate = str(first_event.get("selected_candidate", "none"))
        selected_layer = int(first_event.get("where_layer", -1))
        selected_score_gap_vs_no_growth = score_gap(candidates, selected_candidate, "no_growth")
        pressure_by_layer = first_event.get("pressure_by_layer", {})
        if isinstance(pressure_by_layer, dict) and pressure_by_layer:
            top_pressure_layer = int(max(pressure_by_layer, key=lambda key: float(pressure_by_layer[key])))
            top_growth_name = candidate_name_for_layer(top_pressure_layer, 1)
        top_growth = best_growth_for_layer(candidates, top_pressure_layer)
        if top_growth is not None:
            top_growth_name = str(top_growth.get("name", top_growth_name))
            top_growth_score_gap_vs_no_growth = score_gap(candidates, top_growth_name, "no_growth")
            top_growth_old = float(top_growth.get("old_mean", float("nan")))
            top_growth_c = float(top_growth.get("C_exact", float("nan")))
        if "no_growth" in candidates:
            no_growth_old = float(candidates["no_growth"].get("old_mean", float("nan")))
            no_growth_c = float(candidates["no_growth"].get("C_exact", float("nan")))

    auto_expansions = int(sum(event.how_many_blocks for event in raw_events))
    auto_accepted = int(any(event.accepted for event in raw_events))
    rows = [
        {
            "seed": seed,
            "branch": "old",
            "expansions": 0,
            "growth_accepted": 0,
            **asdict(old_metrics),
            **pressure_summary,
        },
        {
            "seed": seed,
            "branch": "fixed",
            "expansions": 0,
            "growth_accepted": 0,
            **asdict(fixed),
        },
        {
            "seed": seed,
            "branch": "auto_z_controller",
            "expansions": auto_expansions,
            "growth_accepted": auto_accepted,
            "selected_candidate": selected_candidate,
            "selected_layer": selected_layer,
            "top_pressure_layer": top_pressure_layer,
            "selected_score_gap_vs_no_growth": selected_score_gap_vs_no_growth,
            "top_growth_candidate": top_growth_name,
            "top_growth_score_gap_vs_no_growth": top_growth_score_gap_vs_no_growth,
            "top_growth_old_mean": top_growth_old,
            "top_growth_C_exact": top_growth_c,
            "no_growth_old_mean": no_growth_old,
            "no_growth_C_exact": no_growth_c,
            **asdict(auto),
            **pressure_summary,
        },
    ]

    for row in rows:
        print(
            f"[summary] seed={seed} branch={row['branch']} "
            f"A={row['A_exact']:.3f} B={row['B_exact']:.3f} "
            f"C={row['C_exact']:.3f} old={row['old_mean']:.3f} "
            f"expansions={row.get('expansions', 0)}",
            flush=True,
        )
    if first_event is not None:
        print(
            f"[z_governor] selected={selected_candidate} pressure_top={top_pressure_layer} "
            f"selected_gap_vs_no_growth={selected_score_gap_vs_no_growth:.3f} "
            f"top_growth_gap_vs_no_growth={top_growth_score_gap_vs_no_growth:.3f}",
            flush=True,
        )
    return rows, curves, event_rows, pressure_rows


def verdict(all_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    fixed_rows = [row for row in all_rows if row.get("branch") == "fixed"]
    auto_rows = [row for row in all_rows if row.get("branch") == "auto_z_controller"]
    old_rows = [row for row in all_rows if row.get("branch") == "old"]
    if not fixed_rows or not auto_rows:
        return {"status": "FAIL", "passed": False, "reason": "missing fixed or auto rows"}

    mean_fixed_c = float(np.mean([float(row["C_exact"]) for row in fixed_rows]))
    mean_fixed_old = float(np.mean([float(row["old_mean"]) for row in fixed_rows]))
    mean_auto_c = float(np.mean([float(row["C_exact"]) for row in auto_rows]))
    mean_auto_old = float(np.mean([float(row["old_mean"]) for row in auto_rows]))
    mean_old_branch_old = float(np.mean([float(row["old_mean"]) for row in old_rows])) if old_rows else float("nan")
    accepted = int(sum(int(row.get("growth_accepted", 0)) for row in auto_rows))
    expansions = int(sum(int(row.get("expansions", 0)) for row in auto_rows))
    pressure_mode_fraction = float(np.mean([float(row.get("pressure_top_mode_fraction", 0.0)) for row in auto_rows]))
    pressure_gap_ratio = float(np.mean([float(row.get("pressure_mean_gap_ratio", 0.0)) for row in auto_rows]))
    selected_gaps = [float(row.get("selected_score_gap_vs_no_growth", float("nan"))) for row in auto_rows]
    top_growth_gaps = [float(row.get("top_growth_score_gap_vs_no_growth", float("nan"))) for row in auto_rows]
    selected_gap_min = float(np.nanmin(selected_gaps)) if selected_gaps else float("nan")
    top_growth_gap_min = float(np.nanmin(top_growth_gaps)) if top_growth_gaps else float("nan")
    selected_is_pressure_top = [
        int(row.get("selected_layer", -1)) == int(row.get("top_pressure_layer", -2))
        for row in auto_rows
    ]
    pressure_top_selected_rate = float(np.mean(selected_is_pressure_top)) if selected_is_pressure_top else 0.0

    fixed_forgets = mean_fixed_c >= 0.95 and mean_fixed_old <= 0.10
    auto_solves = mean_auto_c >= 0.95 and mean_auto_old >= 0.95
    growth_verified = accepted == len(auto_rows) and expansions >= len(auto_rows)
    pressure_stable = pressure_mode_fraction >= 0.80 and pressure_gap_ratio >= 0.10
    governor_uses_z = pressure_top_selected_rate >= 0.67 and top_growth_gap_min >= 0.20
    branch_beats_no_growth = selected_gap_min >= 0.20

    passed = bool(
        fixed_forgets
        and auto_solves
        and growth_verified
        and pressure_stable
        and governor_uses_z
        and branch_beats_no_growth
    )
    status = "PASS" if passed else "PARTIAL"
    interpretation = (
        "PASS: Z behaved as a stable pressure sensor and the closed-loop governor used it to choose "
        "verified expansion that preserved old skills while acquiring the new skill."
        if passed
        else "PARTIAL: some pieces worked, but the full pressure->governor->Pareto chain did not clear all gates."
    )
    return {
        "status": status,
        "passed": passed,
        "interpretation": interpretation,
        "mean_old_branch_old": mean_old_branch_old,
        "mean_fixed_C_exact": mean_fixed_c,
        "mean_fixed_old": mean_fixed_old,
        "mean_auto_C_exact": mean_auto_c,
        "mean_auto_old": mean_auto_old,
        "accepted_growths": accepted,
        "expansion_blocks": expansions,
        "pressure_top_mode_fraction_mean": pressure_mode_fraction,
        "pressure_gap_ratio_mean": pressure_gap_ratio,
        "pressure_top_selected_rate": pressure_top_selected_rate,
        "selected_score_gap_vs_no_growth_min": selected_gap_min,
        "top_growth_score_gap_vs_no_growth_min": top_growth_gap_min,
        "fixed_forgets": fixed_forgets,
        "auto_solves": auto_solves,
        "growth_verified": growth_verified,
        "pressure_stable": pressure_stable,
        "governor_uses_z": governor_uses_z,
        "branch_beats_no_growth": branch_beats_no_growth,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Z tomography law-candidate toy audit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="1337,2027,31415")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output-dir", default="outputs/z_law_toy")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--n-nums", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--old-steps", type=int, default=900)
    parser.add_argument("--new-steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expansion-lr", type=float, default=1.5e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=150)
    parser.add_argument("--min-expand-step", type=int, default=100)
    parser.add_argument("--new-target-exact", type=float, default=0.82)
    parser.add_argument("--old-drop-trigger", type=float, default=0.08)
    parser.add_argument("--stall-trigger", type=float, default=0.08)
    parser.add_argument("--pressure-scale", type=float, default=10.0)
    parser.add_argument("--pressure-probes", type=int, default=7)
    parser.add_argument("--two-block-severity", type=float, default=1.20)
    parser.add_argument("--max-new-blocks", type=int, default=2)
    parser.add_argument("--governor-probe-steps", type=int, default=80)
    parser.add_argument("--governor-top-k-layers", type=int, default=2)
    parser.add_argument("--governor-old-weight", type=float, default=0.75)
    parser.add_argument("--governor-block-penalty", type=float, default=0.02)
    parser.add_argument("--accept-old-floor", type=float, default=0.85)
    parser.add_argument("--accept-score-slack", type=float, default=0.05)
    return parser


def apply_fast_overrides(args: argparse.Namespace) -> None:
    args.seeds = parse_seeds(args.seeds)[0].__str__()
    args.d_model = min(args.d_model, 64)
    args.d_ff = min(args.d_ff, 128)
    args.old_steps = min(args.old_steps, 250)
    args.new_steps = min(args.new_steps, 300)
    args.batch_size = min(args.batch_size, 64)
    args.probe_batch_size = min(args.probe_batch_size, 48)
    args.eval_samples = min(args.eval_samples, 48)
    args.eval_interval = min(args.eval_interval, 75)
    args.log_interval = min(args.log_interval, 100)
    args.min_expand_step = min(args.min_expand_step, 75)
    args.governor_probe_steps = min(args.governor_probe_steps, 55)
    args.pressure_probes = min(args.pressure_probes, 3)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.fast:
        apply_fast_overrides(args)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Z TOMOGRAPHY LAW-CANDIDATE TOY AUDIT")
    print("=" * 96)
    print(f"device={args.device} seeds={args.seeds}")
    print("claim: Z is a stable pressure sensor used by a branch governor, not a standalone oracle.")
    print("tasks: old=A(copy)+B(reverse), new=C(sort)")
    print("branches: old | fixed_no_growth | auto_z_controller")
    print("stdout is a convenience artifact; CSV/JSON are the paper artifacts.")

    start = time.time()
    all_rows: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    all_events: List[Dict[str, object]] = []
    all_pressure: List[Dict[str, object]] = []

    for seed in parse_seeds(args.seeds):
        rows, curves, events, pressure_rows = run_seed(args, seed)
        all_rows.extend(rows)
        all_curves.extend({"seed": seed, **row} for row in curves)
        all_events.extend(events)
        all_pressure.extend(pressure_rows)

    write_csv(out_dir / "z_law_toy_results.csv", all_rows)
    write_csv(out_dir / "z_law_toy_curves.csv", all_curves)
    write_csv(out_dir / "z_law_toy_pressure.csv", all_pressure)
    (out_dir / "z_law_toy_events.json").write_text(json.dumps(all_events, indent=2, sort_keys=True), encoding="utf-8")

    result = verdict(all_rows)
    result["wall_time_sec"] = float(time.time() - start)
    (out_dir / "z_law_toy_verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 96)
    print("Z LAW TOY VERDICT")
    print("=" * 96)
    print(f"mean_fixed C={result['mean_fixed_C_exact']:.3f} old={result['mean_fixed_old']:.3f}")
    print(f"mean_auto  C={result['mean_auto_C_exact']:.3f} old={result['mean_auto_old']:.3f}")
    print(
        f"pressure_top_mode={result['pressure_top_mode_fraction_mean']:.3f} "
        f"pressure_gap={result['pressure_gap_ratio_mean']:.3f} "
        f"z_selected_rate={result['pressure_top_selected_rate']:.3f}",
        flush=True,
    )
    print(
        f"selected_gap_vs_no_growth_min={result['selected_score_gap_vs_no_growth_min']:.3f} "
        f"top_growth_gap_vs_no_growth_min={result['top_growth_score_gap_vs_no_growth_min']:.3f}",
        flush=True,
    )
    print(str(result["status"]), flush=True)
    print(str(result["interpretation"]), flush=True)
    print(f"wall_time_sec={result['wall_time_sec']:.1f}", flush=True)
    print(f"artifacts={out_dir}", flush=True)


if __name__ == "__main__":
    main()
