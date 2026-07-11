from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ._io import verify_file_manifest
from .acquisition import AcquisitionConfig, PolicyUpdateConfig
from .commit_store import TransactionStore
from .engine import ContinualLearningEngine
from .events import LearningEvent
from .geometry import GeometryController
from .hf_runtime import HuggingFaceContinualRuntime, HuggingFaceRuntimeConfig
from .runtime import TabularContinualRuntime
from .stage1 import (  # Registers two_hop_self.
    write_general_canary_event,
    write_history_profile_events,
    write_stage1_events,
)
from .stream import DirectoryEventSource
from .trajectories import SamplingConfig


def _csv_strings(raw: str) -> Tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _csv_ints(raw: str) -> Tuple[int, ...]:
    return tuple(int(value) for value in _csv_strings(raw))


def _boolean_option(parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    destination = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=destination, action="store_true")
    group.add_argument(f"--no-{name.lstrip('-')}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def _runtime(args: argparse.Namespace):
    if args.runtime == "tabular":
        return TabularContinualRuntime(
            choices=_csv_strings(args.choices),
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
    general_canary: Tuple[Tuple[str, str], ...] = ()
    if args.general_canary:
        payload = json.loads(Path(args.general_canary).read_text(encoding="utf-8"))
        general_canary = tuple((str(item["prompt"]), str(item["target"])) for item in payload)
    config = HuggingFaceRuntimeConfig(
        device=args.device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        target_suffixes=_csv_strings(args.target_suffixes),
        candidate_layers=_csv_ints(args.candidate_layers),
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_gate_init=args.adapter_gate_init,
        acquisition_lr=args.acquisition_lr,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        consolidation_steps=args.consolidation_steps,
        consolidation_lr=args.consolidation_lr,
        consolidation_batch_size=args.consolidation_batch_size,
        consolidation_save_interval=args.consolidation_save_interval,
        profile_rank=args.profile_rank,
        general_canary=general_canary,
        strict_profile_tensors=args.strict_profile_tensors,
        retention_max_drop=args.retention_max_drop,
        general_max_loss_increase=args.general_max_loss_increase,
        stale_rate_max=args.stale_rate_max,
        bootstrap_profile_samples=args.bootstrap_profile_samples,
    )
    return HuggingFaceContinualRuntime(config)


def command_init(args: argparse.Namespace) -> None:
    store = TransactionStore(Path(args.store))
    model_path = Path(args.model)
    if args.runtime == "tabular":
        runtime = _runtime(args)
        runtime.initialize_model(model_path)
    current = store.initialize(model_path=model_path)
    print(json.dumps({"status": "initialized", **current.__dict__}, indent=2, sort_keys=True))


def command_submit(args: argparse.Namespace) -> None:
    source = DirectoryEventSource(Path(args.events))
    submitted = []
    for raw_path in args.event:
        event = LearningEvent.from_dict(json.loads(Path(raw_path).read_text(encoding="utf-8")))
        submitted.append(str(source.submit(event)))
    print(json.dumps({"submitted": submitted}, indent=2, sort_keys=True))


def command_stage1_events(args: argparse.Namespace) -> None:
    paths = write_stage1_events(
        Path(args.manifest),
        Path(args.output_dir),
        acquisition_steps=args.acquisition_steps,
        composition_steps=args.composition_steps,
        group_size=args.group_size,
    )
    print(json.dumps({"events": [str(path) for path in paths]}, indent=2, sort_keys=True))


def command_history_events(args: argparse.Namespace) -> None:
    paths = write_history_profile_events(Path(args.manifest), Path(args.output_dir))
    print(json.dumps({"events": [str(path) for path in paths]}, indent=2, sort_keys=True))


def command_general_canary_event(args: argparse.Namespace) -> None:
    path = write_general_canary_event(
        Path(args.canary),
        Path(args.output),
        seed=args.seed,
    )
    print(json.dumps({"event": str(path)}, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    if args.runtime == "hf" and not args.general_canary:
        raise ValueError("HF continual runs require --general-canary before acquisition starts")
    store = TransactionStore(Path(args.store))
    if not store.current_path.exists():
        if not args.model:
            raise FileNotFoundError("store is uninitialized; pass --model or run init first")
        model_path = Path(args.model)
        runtime = _runtime(args)
        if args.runtime == "tabular":
            runtime.initialize_model(model_path)
        store.initialize(model_path=model_path)
    runtime = _runtime(args)
    acquisition = AcquisitionConfig(
        sampling=SamplingConfig(
            group_size=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        ),
        update=PolicyUpdateConfig(
            kl_coefficient=args.kl_coefficient,
            clip_ratio=args.clip_ratio,
            entropy_coefficient=args.entropy_coefficient,
            grad_clip=args.grad_clip,
        ),
        context_mixture={
            "full": args.context_full,
            "compressed": args.context_compressed,
            "none": args.context_none,
        },
        context_kl_weight=args.context_kl_weight,
        target_anchor_kl=args.target_anchor_kl,
        adaptive_kl=args.adaptive_kl,
        save_interval=args.save_interval,
    )
    engine = ContinualLearningEngine(
        store=store,
        runtime=runtime,
        geometry=GeometryController(
            pressure_coverage=args.pressure_coverage,
            min_layers=args.min_layers,
            max_layers=args.max_layers,
            min_residual_energy=args.min_residual_energy,
            saturation_free_fraction=args.saturation_free_fraction,
            enable_expansion=False,
        ),
        acquisition_config=acquisition,
    )
    source = DirectoryEventSource(Path(args.events))
    source.recover_leases(older_than_seconds=args.lease_timeout)
    result = engine.run_stream(source, max_events=args.max_events)
    result["queue"] = source.checkpoint()
    result["current_version"] = store.current().version
    print(json.dumps(result, indent=2, sort_keys=True))


def command_status(args: argparse.Namespace) -> None:
    store = TransactionStore(Path(args.store))
    source = DirectoryEventSource(Path(args.events)) if args.events else None
    transactions = []
    for path in sorted(store.transactions.glob("**/transaction.json")):
        transactions.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "current": store.current().__dict__ if store.current_path.exists() else None,
        "profiles": store.registry().to_dict() if store.current_path.exists() else None,
        "queue": source.checkpoint() if source else None,
        "transactions": transactions,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    store = TransactionStore(Path(args.store))
    version = store.current().version if args.version < 0 else args.version
    path = store.versions / f"v{version:06d}" / "acceptance.json"
    if not path.exists():
        raise FileNotFoundError(f"version has no acceptance report: {path}")
    print(path.read_text(encoding="utf-8"))


def command_audit(args: argparse.Namespace) -> None:
    store = TransactionStore(Path(args.store))
    issues = []
    if store.current_path.exists():
        store.current()
    for version in sorted(store.versions.glob("v[0-9]*")):
        checksum_path = version / "checksums.json"
        if not checksum_path.exists():
            issues.append(f"version has no checksums: {version}")
            continue
        try:
            verify_file_manifest(
                version,
                json.loads(checksum_path.read_text(encoding="utf-8")),
            )
        except RuntimeError as exc:
            issues.append(f"version checksum failure {version}: {exc}")
    for path in sorted(store.transactions.glob("**/transaction.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") == "running":
            issues.append(f"unfinished transaction: {path.parent}")
        checksum_path = path.parent / "checksums.json"
        if checksum_path.exists():
            try:
                verify_file_manifest(
                    path.parent,
                    json.loads(checksum_path.read_text(encoding="utf-8")),
                    exclude_names=("checksums.json", "transaction.json"),
                )
            except RuntimeError as exc:
                issues.append(f"transaction checksum failure {path.parent}: {exc}")
    access_path = store.root / "data_access.jsonl"
    if access_path.exists():
        for line_number, line in enumerate(access_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid access log line={line_number}: {exc}")
                continue
            if row.get("purpose") == "update" and not row.get("row_ids"):
                issues.append(f"empty update row list at access log line={line_number}")
    payload = {"clean": not issues, "issues": issues, "current": store.current().__dict__}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


def command_plot(args: argparse.Namespace) -> None:
    store = TransactionStore(Path(args.store))
    rows = [row for row in store.journal() if row.get("status") == "committed" and row.get("version", 0) > 0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot requires matplotlib") from exc
    versions = [int(row["version"]) for row in rows]
    labels = [str(row.get("event_key") or row.get("source_event")) for row in rows]
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.4), 3.8))
    axis.plot(versions, versions, marker="o", linewidth=2, color="#1f77b4")
    for version, label in zip(versions, labels):
        axis.annotate(label, (version, version), xytext=(0, 10), textcoords="offset points", ha="center")
    axis.set_xlabel("Committed event")
    axis.set_ylabel("Model version")
    axis.set_title("Continual-learning commit history")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(json.dumps({"plot": str(output), "events": len(rows)}, indent=2))


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime", choices=("tabular", "hf"), default="hf")
    parser.add_argument("--choices", default="A,B")
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--target-suffixes", default="mlp.down_proj,mlp.up_proj")
    parser.add_argument("--candidate-layers", default="")
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--adapter-alpha", type=float, default=32.0)
    parser.add_argument("--adapter-gate-init", type=float, default=-6.0)
    parser.add_argument("--acquisition-lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=160)
    _boolean_option(parser, "--gradient-checkpointing", default=True)
    parser.add_argument("--consolidation-steps", type=int, default=120)
    parser.add_argument("--consolidation-lr", type=float, default=2e-6)
    parser.add_argument("--consolidation-batch-size", type=int, default=1)
    parser.add_argument("--consolidation-save-interval", type=int, default=30)
    parser.add_argument("--profile-rank", type=int, default=8)
    parser.add_argument("--general-canary", default="")
    _boolean_option(parser, "--strict-profile-tensors", default=True)
    parser.add_argument("--retention-max-drop", type=float, default=0.10)
    parser.add_argument("--general-max-loss-increase", type=float, default=0.10)
    parser.add_argument("--stale-rate-max", type=float, default=0.10)
    parser.add_argument("--bootstrap-profile-samples", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Event-driven geometric continual-learning engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Initialize a transactional model store")
    init.add_argument("--store", required=True)
    init.add_argument("--model", required=True)
    add_runtime_arguments(init)
    init.set_defaults(func=command_init)

    submit = subparsers.add_parser("submit", help="Submit one or more frozen LearningEvent files")
    submit.add_argument("--events", required=True)
    submit.add_argument("--event", action="append", required=True)
    submit.set_defaults(func=command_submit)

    stage1 = subparsers.add_parser("stage1-events", help="Convert a frozen Stage-1 manifest into events")
    stage1.add_argument("--manifest", required=True)
    stage1.add_argument("--output-dir", required=True)
    stage1.add_argument("--acquisition-steps", type=int, default=120)
    stage1.add_argument("--composition-steps", type=int, default=120)
    stage1.add_argument("--group-size", type=int, default=4)
    stage1.set_defaults(func=command_stage1_events)

    history = subparsers.add_parser(
        "history-events", help="Convert a frozen prior-skill manifest into profile-only events"
    )
    history.add_argument("--manifest", required=True)
    history.add_argument("--output-dir", required=True)
    history.set_defaults(func=command_history_events)

    general = subparsers.add_parser(
        "general-canary-event",
        help="Freeze a general canary as a profile-only geometry event",
    )
    general.add_argument("--canary", required=True)
    general.add_argument("--output", required=True)
    general.add_argument("--seed", type=int, default=1337)
    general.set_defaults(func=command_general_canary_event)

    run = subparsers.add_parser("run", help="Consume an open-ended event queue")
    run.add_argument("--store", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--model", default="")
    run.add_argument("--max-events", type=int, default=0)
    run.add_argument("--lease-timeout", type=float, default=3600.0)
    run.add_argument("--group-size", type=int, default=4)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--kl-coefficient", type=float, default=0.02)
    run.add_argument("--target-anchor-kl", type=float, default=0.02)
    run.add_argument("--clip-ratio", type=float, default=0.2)
    run.add_argument("--entropy-coefficient", type=float, default=0.0)
    run.add_argument("--grad-clip", type=float, default=0.3)
    run.add_argument("--context-full", type=float, default=0.4)
    run.add_argument("--context-compressed", type=float, default=0.3)
    run.add_argument("--context-none", type=float, default=0.3)
    run.add_argument("--context-kl-weight", type=float, default=1.0)
    _boolean_option(run, "--adaptive-kl", default=True)
    run.add_argument("--save-interval", type=int, default=25)
    run.add_argument("--pressure-coverage", type=float, default=0.8)
    run.add_argument("--min-layers", type=int, default=8)
    run.add_argument("--max-layers", type=int, default=0)
    run.add_argument("--min-residual-energy", type=float, default=0.15)
    run.add_argument("--saturation-free-fraction", type=float, default=0.08)
    add_runtime_arguments(run)
    run.set_defaults(func=command_run)

    status = subparsers.add_parser("status", help="Inspect current version, profiles, queue, and attempts")
    status.add_argument("--store", required=True)
    status.add_argument("--events", default="")
    status.set_defaults(func=command_status)

    evaluate = subparsers.add_parser("evaluate", help="Print a committed version's acceptance report")
    evaluate.add_argument("--store", required=True)
    evaluate.add_argument("--version", type=int, default=-1)
    evaluate.set_defaults(func=command_evaluate)

    audit = subparsers.add_parser("audit", help="Verify committed artifacts and transaction state")
    audit.add_argument("--store", required=True)
    audit.set_defaults(func=command_audit)

    plot = subparsers.add_parser("plot", help="Plot atomic model-version history")
    plot.add_argument("--store", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(func=command_plot)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
