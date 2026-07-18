from __future__ import annotations

import gc
import json
import math
import random
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._io import (
    atomic_write_json,
    file_manifest,
    safe_name,
    sha256_file,
    verify_file_manifest,
)
from .acquisition import PolicyOutput, PolicyStepMetrics, PolicyUpdateConfig
from .artifacts import AcquisitionArtifact, CandidateArtifact
from .commit_store import CurrentVersion, TransactionHandle
from .consolidation import ConsolidationResult, RuntimeEvaluation
from .events import (
    ExampleRecord,
    GateBundle,
    LearningEvent,
    TargetRef,
    VerifierSpec,
    dataset_from_dict,
)
from .evaluator import SystemChecks
from .geometry import GeometryMeasurement, GeometryPlan, LayerMeasurement
from .profiles import ProfileRecord, ProfileRegistry
from .trajectories import PolicyVersion, SamplingConfig, Trajectory, TrainingSample
from .verifiers import build_verifier, score_sync


@dataclass(frozen=True)
class HuggingFaceRuntimeConfig:
    device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    dtype: str = "float32"
    local_files_only: bool = False
    trust_remote_code: bool = True
    target_suffixes: Tuple[str, ...] = ("mlp.down_proj", "mlp.up_proj")
    candidate_layers: Tuple[int, ...] = ()
    adapter_rank: int = 16
    adapter_alpha: float = 32.0
    adapter_gate_init: float = -6.0
    acquisition_lr: float = 2e-5
    max_seq_len: int = 160
    gradient_checkpointing: bool = True
    consolidation_steps: int = 120
    consolidation_lr: float = 2e-6
    consolidation_batch_size: int = 1
    consolidation_save_interval: int = 30
    new_kl_weight: float = 1.0
    old_kl_weight: float = 0.35
    new_hidden_weight: float = 0.05
    old_hidden_weight: float = 0.05
    projection_strength: float = 1.0
    profile_rank: int = 8
    general_canary: Tuple[Tuple[str, str], ...] = ()
    strict_profile_tensors: bool = True
    retention_max_drop: float = 0.10
    general_max_loss_increase: float = 0.10
    stale_rate_max: float = 0.10
    bootstrap_profile_samples: int = 16


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the Hugging Face continual runtime requires torch") from exc
    return torch


def _dtype(name: str):
    torch = _torch()
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported Hugging Face runtime dtype={name}") from exc


def _load_tokenizer(model_id: str, config: HuggingFaceRuntimeConfig):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the Hugging Face continual runtime requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_id: str, device: str, config: HuggingFaceRuntimeConfig):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the Hugging Face continual runtime requires transformers") from exc
    kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_dtype(config.dtype), **kwargs)
    except TypeError:  # transformers 4.x
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype(config.dtype), **kwargs)
    model = model.to(device)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def _freeze(model: Any) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def _infer_layers(model: Any) -> Tuple[int, ...]:
    values = set()
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    for name, _ in model.named_modules():
        match = pattern.search(name)
        if match:
            values.add(int(match.group(1)))
    if not values:
        count = int(getattr(model.config, "num_hidden_layers", 0))
        values.update(range(count))
    if not values:
        raise RuntimeError("could not infer transformer layer indices")
    return tuple(sorted(values))


def _layer_from_name(name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _completion_batch(tokenizer: Any, prompts: Sequence[str], completions: Sequence[str], device: str, max_len: int):
    completion_token_ids = [
        list(tokenizer(str(completion), add_special_tokens=False)["input_ids"])
        for completion in completions
    ]
    return _completion_batch_from_ids(tokenizer, prompts, completion_token_ids, device, max_len)


def _completion_batch_from_ids(
    tokenizer: Any,
    prompts: Sequence[str],
    completion_token_ids: Sequence[Sequence[int]],
    device: str,
    max_len: int,
):
    torch = _torch()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    rows: List[List[int]] = []
    attention: List[List[int]] = []
    completion_masks: List[List[int]] = []
    completion_lengths: List[int] = []
    fixed = max(2, int(max_len))
    for prompt, raw_completion_ids in zip(prompts, completion_token_ids):
        prompt_ids = list(tokenizer(str(prompt), add_special_tokens=False)["input_ids"])
        completion_ids = [int(value) for value in raw_completion_ids]
        if not completion_ids:
            completion_ids = [int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id)]
        completion_ids = completion_ids[: fixed - 1]
        prompt_ids = prompt_ids[-(fixed - len(completion_ids)) :]
        ids = prompt_ids + completion_ids
        pad = fixed - len(ids)
        rows.append([int(pad_id)] * pad + ids)
        attention.append([0] * pad + [1] * len(ids))
        completion_masks.append([0] * (pad + len(prompt_ids)) + [1] * len(completion_ids))
        completion_lengths.append(len(completion_ids))
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(completion_masks, dtype=torch.bool, device=device),
        "completion_lengths": completion_lengths,
    }


def _chosen_logprobs(logits: Any, input_ids: Any):
    torch = _torch()
    logps = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    return logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _stream_tensor(trajectories: Sequence[Trajectory], field_name: str, batch: Mapping[str, Any], device: str):
    torch = _torch()
    mask = batch["completion_mask"][:, 1:]
    output = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    for row_index, trajectory in enumerate(trajectories):
        values = list(getattr(trajectory.sample, field_name, ())) if trajectory.sample else []
        positions = torch.nonzero(mask[row_index], as_tuple=False).flatten()
        if values and len(values) != len(positions):
            if len(values) > len(positions):
                values = values[: len(positions)]
            else:
                raise ValueError(
                    f"trajectory {trajectory.rollout_id} {field_name} has {len(values)} values for "
                    f"{len(positions)} completion tokens"
                )
        if values:
            output[row_index, positions] = torch.tensor(values, dtype=torch.float32, device=device)
    return output


def _masked_mean(values: Any, mask: Any):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


class HuggingFaceTemporaryPolicy:
    def __init__(
        self,
        *,
        model_id: str,
        committed_version: int,
        base_policy_hash: str,
        config: HuggingFaceRuntimeConfig,
        attempt: int = 1,
    ) -> None:
        torch = _torch()
        self.config = config
        self.model_id = model_id
        self.tokenizer = _load_tokenizer(model_id, config)
        self.model = _load_model(model_id, config.device, config)
        self.anchor = _load_model(model_id, config.teacher_device, config)
        _freeze(self.anchor)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        layers = config.candidate_layers or _infer_layers(self.model)
        self._candidate_layers = tuple(layers)
        try:
            from standalone_latent_lora_qwen import LatentLoRAConfig, attach_latent_lora
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HuggingFaceTemporaryPolicy requires standalone_latent_lora_qwen.py from this repository"
            ) from exc
        lora_config = LatentLoRAConfig(
            rank=config.adapter_rank,
            alpha=config.adapter_alpha,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=config.adapter_gate_init,
            freeze_base=True,
        )
        self.attached = attach_latent_lora(
            self.model,
            suffixes=config.target_suffixes,
            layer_indices=set(self._candidate_layers),
            config=lora_config,
        )
        if not self.attached:
            raise RuntimeError(
                f"no adapter modules matched suffixes={config.target_suffixes} layers={self._candidate_layers}"
            )
        params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.acquisition_lr, foreach=False)
        self._version = PolicyVersion(committed_version, attempt, 0)
        self._base_policy_hash = base_policy_hash

    @property
    def version(self) -> PolicyVersion:
        return self._version

    @property
    def base_policy_hash(self) -> str:
        return self._base_policy_hash

    @property
    def candidate_layers(self) -> Sequence[int]:
        return self._candidate_layers

    def generate(self, prompt: str, sampling: SamplingConfig, *, seed: int) -> PolicyOutput:
        torch = _torch()
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        self.model.eval()
        kwargs: Dict[str, Any] = {
            **encoded,
            "do_sample": True,
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "max_new_tokens": int(sampling.max_new_tokens),
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        cuda_devices = []
        if str(self.config.device).startswith("cuda"):
            device_index = torch.device(self.config.device).index
            cuda_devices = [device_index if device_index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed(int(seed))
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                generated = self.model.generate(**kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        completion_ids = generated.sequences[0, prompt_width:]
        scores = list(generated.scores)
        logprobs: List[float] = []
        entropies: List[float] = []
        for index, score in enumerate(scores[: len(completion_ids)]):
            distribution = torch.nn.functional.log_softmax(score[0].float(), dim=-1)
            token_id = int(completion_ids[index])
            logprobs.append(float(distribution[token_id].item()))
            probabilities = distribution.exp()
            entropies.append(float((-(probabilities * distribution).sum()).item()))
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if not completion_ids.numel():
            fallback = int(self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0)
            completion_ids = torch.tensor([fallback], device=self.config.device)
            logprobs = [0.0]
            entropies = [0.0]
        sample = TrainingSample(
            token_ids=[int(value) for value in completion_ids.tolist()],
            completion_mask=[True] * len(completion_ids),
            rollout_logprobs=logprobs,
        )
        return PolicyOutput(text, sample, sum(entropies) / len(entropies), {"backend": "huggingface"})

    def supervised_sample(self, prompt: str, target: str) -> TrainingSample:
        batch = _completion_batch(self.tokenizer, [prompt], [target], self.config.device, self.config.max_seq_len)
        self.model.eval()
        torch = _torch()
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logps = _chosen_logprobs(outputs.logits, batch["input_ids"])
        mask = batch["completion_mask"][:, 1:]
        values = logps[mask].detach().float().cpu().tolist()
        token_ids = batch["input_ids"][:, 1:][mask].detach().cpu().tolist()
        return TrainingSample(token_ids=token_ids, completion_mask=[True] * len(token_ids), rollout_logprobs=values)

    def reference_logprobs(self, prompt: str, completion: str, *, token_ids: Sequence[int]) -> Sequence[float]:
        torch = _torch()
        batch = _completion_batch_from_ids(
            self.tokenizer, [prompt], [token_ids], self.config.device, self.config.max_seq_len
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            values = _chosen_logprobs(outputs.logits, batch["input_ids"])[
                batch["completion_mask"][:, 1:]
            ].float().cpu().tolist()
        if len(values) != len(token_ids):
            values = values[-len(token_ids) :]
        return values

    def update(self, trajectories: Sequence[Trajectory], config: PolicyUpdateConfig) -> PolicyStepMetrics:
        torch = _torch()
        if not trajectories:
            raise ValueError("Hugging Face policy update requires trajectories")
        prompts = [item.prompt for item in trajectories]
        batch = _completion_batch_from_ids(
            self.tokenizer,
            prompts,
            [item.sample.token_ids for item in trajectories],
            self.config.device,
            self.config.max_seq_len,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        current = _chosen_logprobs(outputs.logits, batch["input_ids"])
        mask = batch["completion_mask"][:, 1:].float()
        old_logps = _stream_tensor(trajectories, "rollout_logprobs", batch, self.config.device)
        advantages = _stream_tensor(trajectories, "advantages", batch, self.config.device)
        ce_weights = _stream_tensor(trajectories, "ce_weights", batch, self.config.device)
        ref_weights = _stream_tensor(trajectories, "reference_kl_weights", batch, self.config.device)
        reference = _stream_tensor(trajectories, "reference_logprobs", batch, self.config.device)

        ratio = torch.exp((current.float() - old_logps).clamp(-20, 20))
        clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
        surrogate = torch.minimum(ratio * advantages, clipped * advantages)
        policy_mask = mask * (advantages != 0).float()
        policy_loss = -_masked_mean(surrogate, policy_mask)
        ce_mask = mask * (ce_weights != 0).float()
        ce_loss = -_masked_mean(current.float() * ce_weights, ce_mask)
        ref_advantage = (reference - old_logps).detach()
        ref_mask = mask * (ref_weights != 0).float()
        reference_loss = -_masked_mean(current.float() * ref_advantage * ref_weights, ref_mask)

        anchor_batch = {
            key: value.to(self.config.teacher_device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask"}
        }
        self.anchor.eval()
        with torch.no_grad():
            anchor_outputs = self.anchor(**anchor_batch, use_cache=False)
        student_log_distribution = torch.nn.functional.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        anchor_log_distribution = torch.nn.functional.log_softmax(
            anchor_outputs.logits[:, :-1, :].to(self.config.device).float(), dim=-1
        )
        token_kl = torch.nn.functional.kl_div(
            student_log_distribution,
            anchor_log_distribution,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        anchor_kl = _masked_mean(token_kl, mask)
        probabilities = student_log_distribution.exp()
        token_entropy = -(probabilities * student_log_distribution).sum(dim=-1)
        entropy = _masked_mean(token_entropy, mask)
        total = (
            policy_loss
            + ce_loss
            + reference_loss
            + float(config.kl_coefficient) * anchor_kl
            - float(config.entropy_coefficient) * entropy
        )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite temporary policy loss")
        total.backward()
        params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        for parameter in params:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite temporary policy gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip))
        self.optimizer.step()
        self._version = PolicyVersion(
            self._version.committed,
            self._version.attempt,
            self._version.update + 1,
        )
        return PolicyStepMetrics(
            loss=float(total.item()),
            policy_loss=float(policy_loss.item()),
            ce_loss=float(ce_loss.item()),
            reference_kl=float(reference_loss.item()),
            anchor_kl=float(anchor_kl.item()),
            entropy=float(entropy.item()),
            grad_norm=float(grad_norm),
        )

    def save_temporary(self, path: Path) -> None:
        torch = _torch()
        path.mkdir(parents=True, exist_ok=True)
        trainable = {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(trainable, path / "adapter.pt")
        atomic_write_json(
            path / "adapter_config.json",
            {
                "model_id": self.model_id,
                "version": asdict(self._version),
                "base_policy_hash": self._base_policy_hash,
                "candidate_layers": self._candidate_layers,
                "runtime": asdict(self.config),
            },
        )

    def save_resume(self, path: Path) -> None:
        torch = _torch()
        self.save_temporary(path)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "version": asdict(self._version),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(),
            },
            path / "resume.pt",
        )

    def load_resume(self, path: Path) -> None:
        torch = _torch()
        self.load_adapter_state(path)
        payload = torch.load(path / "resume.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(payload["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.config.device)
        self._version = PolicyVersion(**payload["version"])
        torch.set_rng_state(payload["torch_cpu_rng"])
        if torch.cuda.is_available() and payload.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
        random.setstate(payload["python_rng"])

    def load_adapter_state(self, path: Path) -> None:
        torch = _torch()
        state = torch.load(path / "adapter.pt", map_location=self.config.device, weights_only=False)
        parameters = dict(self.model.named_parameters())
        missing = sorted(set(state) - set(parameters))
        if missing:
            raise KeyError(f"adapter state contains unknown parameters={missing[:5]}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))

    def adapter_pressure(self) -> Dict[int, float]:
        pressure: Dict[int, float] = {}
        for name, delta in self.adapter_deltas().items():
            layer = _layer_from_name(name)
            if layer is None:
                continue
            pressure[layer] = pressure.get(layer, 0.0) + float(delta.norm().item())
        return pressure

    def adapter_deltas(self) -> Dict[str, Any]:
        deltas: Dict[str, Any] = {}
        for name, module in self.attached:
            if module.lora_A is None or module.lora_B is None:
                continue
            delta = module.lora_B.detach().float() @ module.lora_A.detach().float()
            delta = delta * float(module.scaling) * float(module.gate_value().detach().float())
            deltas[f"{name}.weight"] = delta.cpu().contiguous()
        return deltas

    def move_teacher_to_auxiliary(self) -> None:
        self.model = self.model.to(self.config.teacher_device)

    def close(self) -> None:
        del self.model, self.anchor, self.optimizer
        gc.collect()
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HuggingFaceContinualRuntime:
    def __init__(self, config: Optional[HuggingFaceRuntimeConfig] = None) -> None:
        self.config = config or HuggingFaceRuntimeConfig()
        self._temporary: Dict[str, HuggingFaceTemporaryPolicy] = {}
        self._candidates: Dict[str, Any] = {}

    def open_temporary_policy(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        transaction: TransactionHandle,
    ) -> HuggingFaceTemporaryPolicy:
        policy = HuggingFaceTemporaryPolicy(
            model_id=current.model_path,
            committed_version=current.version,
            base_policy_hash=current.commit_hash,
            config=self.config,
        )
        self._temporary[transaction.attempt_id] = policy
        return policy

    def profile_existing(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        model = _load_model(current.model_path, self.config.device, self.config)
        tokenizer = _load_tokenizer(current.model_path, self.config)
        layers = self.config.candidate_layers or _infer_layers(model)
        store_root = transaction.root.parents[2]
        blob_root = store_root / "profile_blobs"
        blob_root.mkdir(parents=True, exist_ok=True)
        tensor_path, tensor_hash, effective_rank, selected_layers = _write_existing_gradient_profile(
            model,
            tokenizer,
            event,
            blob_root,
            layers,
            self.config,
        )
        records = (event.eval_examples or event.examples).records
        metrics = _evaluate_records(
            model,
            tokenizer,
            records,
            event,
            self.config.device,
            self.config.max_seq_len,
        )
        candidate_registry = registry.clone()
        profile_id = f"capability:{event.event_id}:r{event.revision}"
        canary_root = store_root / "canaries"
        canary_root.mkdir(parents=True, exist_ok=True)
        canary_path = canary_root / f"{safe_name(profile_id)}-{event.fingerprint[:12]}.json"
        atomic_write_json(
            canary_path,
            {
                "dataset": asdict(event.eval_examples or event.examples),
                "verifier": asdict(event.verifier or VerifierSpec("exact_match")),
            },
        )
        candidate_registry.register(
            ProfileRecord(
                profile_id=profile_id,
                capability=event.event_id,
                dependencies=tuple(event.dependencies),
                selected_layers=tuple(selected_layers),
                checkpoint_version=current.version + 1,
                checkpoint_hash=current.commit_hash,
                creation_event=event.event_key,
                scope={"profile_only": "true"},
                tensor_path=str(tensor_path),
                tensor_hash=tensor_hash,
                tensor_schema="gradient_basis_v1",
                metrics={
                    "effective_rank": float(effective_rank),
                    "baseline_capability": float(metrics["capability"]),
                },
                canary_dataset=(event.eval_examples or event.examples).dataset_id,
                canary_path=str(canary_path),
            )
        )
        registry_path = transaction.root / "candidate" / "registry.json"
        candidate_registry.save(registry_path)
        metrics_path = transaction.root / "candidate" / "profile_metrics.json"
        atomic_write_json(metrics_path, metrics)
        candidate = CandidateArtifact(
            event_key=event.event_key,
            model_path=current.model_path,
            registry_path=str(registry_path),
            metrics_path=str(metrics_path),
            source_policy_version=current.version,
            selected_layers=tuple(selected_layers),
            profile_ids=(profile_id,),
            numerical_stable=_model_finite(model),
            metadata={"runtime": "huggingface_local", "reuse_current_model": True, "profile_only": True},
        )
        self._candidates[transaction.attempt_id] = model
        return ConsolidationResult(candidate, candidate_registry)

    def measure_geometry(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> GeometryMeasurement:
        policy = self._temporary_policy(transaction, current, acquisition)
        active = registry.active()
        profile_bases = _load_profile_bases(
            active,
            current_model_path=Path(current.model_path),
            strict=self.config.strict_profile_tensors,
        )
        measurements = _measure_adapter_geometry(
            policy.adapter_deltas(),
            acquisition.candidate_layers,
            profile_bases,
            supersedes=event.supersedes,
        )
        return GeometryMeasurement(
            event.event_key,
            measurements,
            current.commit_hash,
            acquisition.fingerprint,
        )

    def consolidate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        plan: GeometryPlan,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        torch = _torch()
        policy = self._temporary_policy(transaction, current, acquisition)
        policy.move_teacher_to_auxiliary()
        new_teacher = policy.model
        old_anchor = policy.anchor
        resume_root = transaction.root / "consolidation_resume"
        resume_state_path = resume_root / "resume.pt"
        start_step = 0
        losses: List[float] = []
        if resume_state_path.exists():
            checksums_path = resume_root / "checksums.json"
            expected = json.loads(checksums_path.read_text(encoding="utf-8"))
            verify_file_manifest(resume_root, expected)
            resume_payload = torch.load(resume_state_path, map_location="cpu", weights_only=False)
            if resume_payload.get("event_fingerprint") != event.fingerprint:
                raise RuntimeError("consolidation resume artifact belongs to a different event")
            if int(resume_payload.get("base_version", -1)) != current.version:
                raise RuntimeError("consolidation resume artifact belongs to a stale base version")
            student = _load_model(str(resume_root / "model"), self.config.device, self.config)
            start_step = int(resume_payload["step"])
            losses = [float(value) for value in resume_payload.get("losses", ())]
        else:
            student = _load_model(current.model_path, self.config.device, self.config)
        for parameter in student.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=self.config.consolidation_lr,
            foreach=False,
        )
        if resume_state_path.exists():
            optimizer.load_state_dict(resume_payload["optimizer"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(self.config.device)
        rows = [
            json.loads(line)
            for line in Path(acquisition.sample_ledger_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise RuntimeError("consolidation received an empty acquisition ledger")
        protected = [
            registry.records[profile_id]
            for profile_id in plan.protected_profile_ids
            if profile_id in registry.records
        ]
        selected_layers = tuple(plan.selected_layers)
        projected_modules = 0
        for step in range(start_step + 1, self.config.consolidation_steps + 1):
            step_rng = random.Random(event.seed + 17_171 + 104729 * step)
            selected = [
                rows[step_rng.randrange(len(rows))]
                for _ in range(self.config.consolidation_batch_size)
            ]
            prompts = [str(item["prompt"]) for item in selected]
            completion_ids = [
                list(item.get("sample", {}).get("token_ids", ()))
                or list(policy.tokenizer(str(item["completion"]), add_special_tokens=False)["input_ids"])
                for item in selected
            ]
            student_batch = _completion_batch_from_ids(
                policy.tokenizer, prompts, completion_ids, self.config.device, self.config.max_seq_len
            )
            teacher_batch = _completion_batch_from_ids(
                policy.tokenizer, prompts, completion_ids, self.config.teacher_device, self.config.max_seq_len
            )
            optimizer.zero_grad(set_to_none=True)
            student_outputs = student(
                input_ids=student_batch["input_ids"],
                attention_mask=student_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            with torch.no_grad():
                old_outputs = old_anchor(
                    input_ids=teacher_batch["input_ids"],
                    attention_mask=teacher_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                # Crucially, the acquired teacher is evaluated on the ordinary deployment prompt.
                new_outputs = new_teacher(
                    input_ids=teacher_batch["input_ids"],
                    attention_mask=teacher_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            mask = student_batch["completion_mask"][:, 1:].float()
            student_logps = torch.nn.functional.log_softmax(student_outputs.logits[:, :-1, :].float(), dim=-1)
            old_logps = torch.nn.functional.log_softmax(
                old_outputs.logits[:, :-1, :].to(self.config.device).float(), dim=-1
            )
            new_logps = torch.nn.functional.log_softmax(
                new_outputs.logits[:, :-1, :].to(self.config.device).float(), dim=-1
            )
            new_kl = _masked_mean(
                torch.nn.functional.kl_div(student_logps, new_logps, reduction="none", log_target=True).sum(-1),
                mask,
            )
            old_kl = _masked_mean(
                torch.nn.functional.kl_div(student_logps, old_logps, reduction="none", log_target=True).sum(-1),
                mask,
            )
            new_hidden = _hidden_alignment(student_outputs.hidden_states, new_outputs.hidden_states, selected_layers)
            old_hidden = _hidden_alignment(student_outputs.hidden_states, old_outputs.hidden_states, selected_layers)
            loss = (
                self.config.new_kl_weight * new_kl
                + self.config.old_kl_weight * old_kl
                + self.config.new_hidden_weight * new_hidden
                + self.config.old_hidden_weight * old_hidden
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite consolidation loss at step={step}")
            loss.backward()
            projected_modules = _project_profile_gradients(
                student,
                protected,
                current_model_path=Path(current.model_path),
                strength=self.config.projection_strength,
                strict=self.config.strict_profile_tensors,
            )
            parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
            for parameter in parameters:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite consolidation gradient at step={step}")
            torch.nn.utils.clip_grad_norm_(parameters, 0.3)
            optimizer.step()
            losses.append(float(loss.item()))
            if (
                self.config.consolidation_save_interval > 0
                and step % self.config.consolidation_save_interval == 0
                and step < self.config.consolidation_steps
            ):
                _save_consolidation_resume(
                    resume_root,
                    student,
                    policy.tokenizer,
                    optimizer,
                    step=step,
                    losses=losses,
                    event_fingerprint=event.fingerprint,
                    base_version=current.version,
                )

        for name, parameter in student.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"candidate model contains non-finite parameter={name}")
        candidate_model = transaction.candidate_model_dir
        if candidate_model.exists():
            shutil.rmtree(candidate_model)
        candidate_model.mkdir(parents=True)
        student.save_pretrained(candidate_model, safe_serialization=True)
        policy.tokenizer.save_pretrained(candidate_model)
        current_model_path = Path(current.model_path)
        if current_model_path.is_dir():
            for auxiliary_name in ("geometry_profiles",):
                source = current_model_path / auxiliary_name
                destination = candidate_model / auxiliary_name
                if source.exists() and not destination.exists():
                    shutil.copytree(source, destination)
        candidate_registry = registry.clone()
        if plan.release_profile_ids:
            released = candidate_registry.release_closure(event.supersedes)
            candidate_registry.retire(released)
        tensor_path, tensor_hash, effective_rank = _write_delta_profile(
            student,
            old_anchor,
            candidate_model,
            selected_layers,
            self.config,
            seed=event.seed,
        )
        profile_id = f"capability:{event.event_id}:r{event.revision}"
        store_root = transaction.root.parents[2]
        canary_dir = store_root / "canaries"
        canary_dir.mkdir(parents=True, exist_ok=True)
        canary_path = canary_dir / f"{safe_name(profile_id)}-{event.fingerprint[:12]}.json"
        canary_dataset = event.eval_examples or event.examples
        canary_verifier = event.verifier or VerifierSpec("exact_match")
        atomic_write_json(
            canary_path,
            {"dataset": asdict(canary_dataset), "verifier": asdict(canary_verifier)},
        )
        candidate_registry.register(
            ProfileRecord(
                profile_id=profile_id,
                capability=event.event_id,
                dependencies=tuple(event.dependencies),
                selected_layers=selected_layers,
                checkpoint_version=current.version + 1,
                checkpoint_hash=acquisition.fingerprint,
                creation_event=event.event_key,
                scope={"event_kind": event.kind},
                tensor_path=tensor_path,
                tensor_hash=tensor_hash,
                tensor_schema="parameter_delta_basis_v1",
                metrics={
                    "effective_rank": float(effective_rank),
                    "mean_consolidation_loss": sum(losses) / max(1, len(losses)),
                    "projected_modules": float(projected_modules),
                },
                canary_dataset=canary_dataset.dataset_id,
                canary_path=str(canary_path),
            )
        )
        registry_path = transaction.root / "candidate" / "registry.json"
        candidate_registry.save(registry_path)
        metrics_path = transaction.root / "candidate" / "consolidation_metrics.json"
        atomic_write_json(
            metrics_path,
            {
                "steps": self.config.consolidation_steps,
                "mean_loss": sum(losses) / max(1, len(losses)),
                "projected_modules": projected_modules,
                "selected_layers": selected_layers,
            },
        )
        candidate = CandidateArtifact(
            event_key=event.event_key,
            model_path=str(candidate_model),
            registry_path=str(registry_path),
            metrics_path=str(metrics_path),
            source_policy_version=current.version,
            selected_layers=selected_layers,
            profile_ids=(profile_id,),
            numerical_stable=True,
            metadata={"runtime": "huggingface_local", "label_free_reward_consolidation": True},
        )
        self._candidates[transaction.attempt_id] = student
        if resume_root.exists():
            shutil.rmtree(resume_root)
        return ConsolidationResult(candidate, candidate_registry)

    def evaluate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        result: ConsolidationResult,
        transaction: TransactionHandle,
    ) -> RuntimeEvaluation:
        policy = self._temporary.get(transaction.attempt_id)
        tokenizer = policy.tokenizer if policy is not None else _load_tokenizer(current.model_path, self.config)
        candidate_model = self._candidates.get(transaction.attempt_id)
        if candidate_model is None:
            candidate_model = _load_model(result.candidate.model_path, self.config.device, self.config)
        baseline_model = policy.anchor if policy is not None else _load_model(
            current.model_path, self.config.teacher_device, self.config
        )
        records = (event.eval_examples or event.examples).records
        candidate_metrics = _evaluate_records(
            candidate_model, tokenizer, records, event, self.config.device, self.config.max_seq_len
        )
        baseline_metrics = _evaluate_records(
            baseline_model, tokenizer, records, event, self.config.teacher_device, self.config.max_seq_len
        )
        if self.config.general_canary:
            candidate_metrics["general_loss"] = _teacher_forced_loss(
                candidate_model,
                tokenizer,
                self.config.general_canary,
                self.config.device,
                self.config.max_seq_len,
            )
            baseline_metrics["general_loss"] = _teacher_forced_loss(
                baseline_model,
                tokenizer,
                self.config.general_canary,
                self.config.teacher_device,
                self.config.max_seq_len,
            )
            candidate_metrics["general_canary_configured"] = 1.0
            baseline_metrics["general_canary_configured"] = 1.0
        else:
            candidate_metrics["general_canary_configured"] = 0.0
            baseline_metrics["general_canary_configured"] = 0.0
        numerical = _model_finite(candidate_model)
        retention_stable = True
        retention_details: Dict[str, Any] = {}
        for profile_id, profile in sorted(result.registry.records.items()):
            if profile.status != "protected" or profile.creation_event == event.event_key:
                continue
            metric_key = f"retention:{profile_id}"
            if not profile.canary_path or "baseline_capability" not in profile.metrics:
                retention_stable = False
                retention_details[profile_id] = "missing canary or baseline"
                continue
            raw_canary_path = Path(profile.canary_path)
            canary_path = (
                raw_canary_path
                if raw_canary_path.is_absolute()
                else Path(result.candidate.model_path) / raw_canary_path
            )
            if not canary_path.exists():
                retention_stable = False
                retention_details[profile_id] = f"missing canary file {canary_path}"
                continue
            payload = json.loads(canary_path.read_text(encoding="utf-8"))
            dataset = dataset_from_dict(payload["dataset"])
            verifier_spec = VerifierSpec(**payload["verifier"])
            canary_event = LearningEvent(
                event_id=f"canary:{profile_id}",
                revision=0,
                kind="reward",
                examples=dataset,
                targets=TargetRef(visibility="verifier_only"),
                verifier=verifier_spec,
                gates=GateBundle(
                    require_retention=False,
                    require_general_capability=False,
                    require_staleness_on_revision=False,
                ),
            )
            score = _evaluate_records(
                candidate_model,
                tokenizer,
                dataset.records,
                canary_event,
                self.config.device,
                self.config.max_seq_len,
            )["capability"]
            baseline_score = float(profile.metrics["baseline_capability"])
            candidate_metrics[metric_key] = score
            baseline_metrics[metric_key] = baseline_score
            drop = baseline_score - score
            retention_details[profile_id] = {"baseline": baseline_score, "candidate": score, "drop": drop}
            if drop > self.config.retention_max_drop:
                retention_stable = False
        new_profile_id = f"capability:{event.event_id}:r{event.revision}"
        if new_profile_id in result.registry.records:
            profile = result.registry.records[new_profile_id]
            result.registry.records[new_profile_id] = replace(
                profile,
                metrics={**dict(profile.metrics), "baseline_capability": float(candidate_metrics["capability"])},
            )
            result.registry.save(Path(result.candidate.registry_path))
        general_stable = bool(self.config.general_canary) and (
            float(candidate_metrics.get("general_loss", math.inf))
            - float(baseline_metrics.get("general_loss", math.inf))
            <= self.config.general_max_loss_increase
        )
        staleness_clean = float(candidate_metrics.get("stale_rate", 0.0)) <= self.config.stale_rate_max
        return RuntimeEvaluation(
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            checks=SystemChecks(
                numerical_stable=numerical,
                access_audit_clean=True,
                within_budget=True,
                details={"runtime": "huggingface_local", "retention": retention_details},
                retention_stable=retention_stable,
                general_stable=general_stable,
                staleness_clean=staleness_clean,
            ),
        )

    def _temporary_policy(
        self,
        transaction: TransactionHandle,
        current: CurrentVersion,
        acquisition: AcquisitionArtifact,
    ) -> HuggingFaceTemporaryPolicy:
        policy = self._temporary.get(transaction.attempt_id)
        if policy is not None:
            return policy
        policy = HuggingFaceTemporaryPolicy(
            model_id=current.model_path,
            committed_version=current.version,
            base_policy_hash=current.commit_hash,
            config=self.config,
        )
        policy.load_adapter_state(Path(acquisition.adapter_path))
        self._temporary[transaction.attempt_id] = policy
        return policy


def _hidden_alignment(student_hidden: Sequence[Any], teacher_hidden: Sequence[Any], layers: Sequence[int]):
    torch = _torch()
    losses = []
    for layer in layers:
        index = int(layer) + 1
        if index >= len(student_hidden) or index >= len(teacher_hidden):
            continue
        student = student_hidden[index].float()
        teacher = teacher_hidden[index].to(student.device).float()
        losses.append(torch.nn.functional.mse_loss(student, teacher))
    if not losses:
        return student_hidden[0].sum() * 0.0
    return sum(losses) / len(losses)


def _project_profile_gradients(
    model: Any,
    profiles: Sequence[ProfileRecord],
    *,
    current_model_path: Path,
    strength: float,
    strict: bool,
) -> int:
    torch = _torch()
    if not profiles or strength <= 0:
        return 0
    profile_bases = _load_profile_bases(
        profiles,
        current_model_path=current_model_path,
        strict=strict,
    )
    basis_by_name: Dict[str, List[Any]] = {}
    for parameters in profile_bases.values():
        for parameter_name, basis in parameters.items():
            basis_by_name.setdefault(parameter_name, []).append(basis)
    projected = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in basis_by_name or parameter.grad.ndim != 2:
            continue
        gradient = parameter.grad
        compatible = [
            raw_basis.to(gradient.device, gradient.dtype)
            for raw_basis in basis_by_name[name]
            if raw_basis.shape[0] == gradient.shape[0]
        ]
        if compatible:
            basis = _orthonormal_basis(torch.cat(compatible, dim=1))
            if basis.shape[1]:
                gradient.sub_(float(strength) * basis @ (basis.transpose(0, 1) @ gradient))
                projected += 1
    return projected


def _load_profile_bases(
    profiles: Sequence[ProfileRecord],
    *,
    current_model_path: Path,
    strict: bool,
) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        if not profile.tensor_path:
            if strict:
                raise RuntimeError(f"protected profile has no tensor payload: {profile.profile_id}")
            continue
        raw_tensor_path = Path(profile.tensor_path)
        tensor_path = raw_tensor_path if raw_tensor_path.is_absolute() else current_model_path / raw_tensor_path
        metadata_path = tensor_path.with_suffix(".json")
        if not tensor_path.exists() or not metadata_path.exists():
            if strict:
                raise FileNotFoundError(f"protected profile tensor payload is missing: {tensor_path}")
            continue
        if profile.tensor_hash and sha256_file(tensor_path) != profile.tensor_hash:
            raise RuntimeError(f"protected profile tensor checksum mismatch: {profile.profile_id}")
        from safetensors.torch import load_file

        tensors = load_file(str(tensor_path), device="cpu")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded[profile.profile_id] = {
            str(parameter_name): tensors[key].float().cpu()
            for key, parameter_name in metadata["parameters"].items()
        }
    return loaded


def _measure_adapter_geometry(
    deltas: Mapping[str, Any],
    candidate_layers: Sequence[int],
    profile_bases: Mapping[str, Mapping[str, Any]],
    *,
    supersedes: Sequence[str] = (),
) -> Tuple[LayerMeasurement, ...]:
    torch = _torch()
    stale = set(supersedes)
    measurements: List[LayerMeasurement] = []
    for layer in candidate_layers:
        layer_deltas = {
            name: delta.detach().float().cpu()
            for name, delta in deltas.items()
            if _layer_from_name(name) == int(layer) and delta.ndim == 2
        }
        total_energy = sum(float(delta.square().sum().item()) for delta in layer_deltas.values())
        dimension = sum(int(delta.shape[0]) for delta in layer_deltas.values()) or 1
        overlap_energy: Dict[str, float] = {profile_id: 0.0 for profile_id in profile_bases}
        residual_energy = 0.0
        occupied_rank = 0
        for parameter_name, delta in layer_deltas.items():
            all_bases = []
            for profile_id, parameters in profile_bases.items():
                raw_basis = parameters.get(parameter_name)
                if raw_basis is None or raw_basis.ndim != 2 or raw_basis.shape[0] != delta.shape[0]:
                    continue
                basis = _orthonormal_basis(raw_basis.float())
                if not basis.shape[1]:
                    continue
                projected = basis @ (basis.transpose(0, 1) @ delta)
                overlap_energy[profile_id] += float(projected.square().sum().item())
                all_bases.append(basis)
            if all_bases:
                occupied = _orthonormal_basis(torch.cat(all_bases, dim=1))
                occupied_rank += int(occupied.shape[1])
                residual = delta - occupied @ (occupied.transpose(0, 1) @ delta)
            else:
                residual = delta
            residual_energy += float(residual.square().sum().item())
        occupied_rank = min(occupied_rank, dimension)
        overlaps = {
            profile_id: min(1.0, energy / max(total_energy, 1e-12))
            for profile_id, energy in overlap_energy.items()
            if energy > 0.0
        }
        conflicts = {
            profile_id: 1.0 if profile_id in stale else overlap
            for profile_id, overlap in overlaps.items()
        }
        for profile_id in stale:
            if profile_id in profile_bases:
                conflicts[profile_id] = 1.0
        measurements.append(
            LayerMeasurement(
                layer=int(layer),
                pressure=math.sqrt(max(0.0, total_energy)),
                residual_energy=(residual_energy / total_energy) if total_energy > 0 else 1.0,
                occupied_rank=occupied_rank,
                dimension=dimension,
                profile_overlaps=overlaps,
                directional_conflicts=conflicts,
            )
        )
    return tuple(measurements)


def _orthonormal_basis(matrix: Any):
    torch = _torch()
    if matrix.ndim != 2:
        raise ValueError("profile basis must be a matrix")
    if matrix.numel() == 0:
        return matrix[:, :0]
    left, singular_values, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    if not singular_values.numel():
        raise ValueError("profile basis has no singular values")
    tolerance = (
        torch.finfo(singular_values.dtype).eps
        * max(matrix.shape)
        * float(singular_values.max().item())
    )
    rank = int((singular_values > tolerance).sum().item())
    return left[:, :rank].to(matrix.device, matrix.dtype)


def _write_delta_profile(
    student: Any,
    old_anchor: Any,
    candidate_model: Path,
    selected_layers: Sequence[int],
    config: HuggingFaceRuntimeConfig,
    *,
    seed: int,
) -> Tuple[str, str, int]:
    torch = _torch()
    old_parameters = dict(old_anchor.named_parameters())
    tensors: Dict[str, Any] = {}
    names: Dict[str, str] = {}
    total_rank = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for name, parameter in student.named_parameters():
        layer = _layer_from_name(name)
        if layer not in selected_layers or parameter.ndim != 2 or not name.endswith(".weight"):
            continue
        if not any(name.endswith(f"{suffix}.weight") for suffix in config.target_suffixes):
            continue
        old = old_parameters.get(name)
        if old is None:
            continue
        delta = parameter.detach().float().cpu() - old.detach().float().cpu()
        rank = min(config.profile_rank, delta.shape[0], delta.shape[1])
        if rank <= 0 or float(delta.norm()) == 0.0:
            continue
        omega = torch.randn(delta.shape[1], rank, generator=generator)
        basis, _ = torch.linalg.qr(delta @ omega, mode="reduced")
        key = f"basis_{len(tensors):04d}"
        tensors[key] = basis.contiguous()
        names[key] = name
        total_rank += basis.shape[1]
    if not tensors:
        raise RuntimeError("consolidation produced no non-zero selected-module profile tensors")
    from safetensors.torch import save_file

    profile_dir = candidate_model / "geometry_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    provisional = profile_dir / "profile.tmp.safetensors"
    save_file(tensors, str(provisional))
    digest = sha256_file(provisional)
    final = profile_dir / f"{digest}.safetensors"
    provisional.replace(final)
    atomic_write_json(final.with_suffix(".json"), {"schema": "parameter_delta_basis_v1", "parameters": names})
    return str(final.relative_to(candidate_model)), digest, total_rank


def _write_existing_gradient_profile(
    model: Any,
    tokenizer: Any,
    event: LearningEvent,
    blob_root: Path,
    candidate_layers: Sequence[int],
    config: HuggingFaceRuntimeConfig,
) -> Tuple[Path, str, int, Tuple[int, ...]]:
    torch = _torch()
    rows = list(event.examples.records)[: max(1, int(config.bootstrap_profile_samples))]
    if not rows or any(record.target is None for record in rows):
        raise ValueError("profile-only events require private targets for gradient profiling")
    for parameter in model.parameters():
        parameter.requires_grad = True
    basis_parts: Dict[str, List[Any]] = {}
    layer_pressure: Dict[int, float] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(event.seed))
    selected_set = set(int(value) for value in candidate_layers)
    for record in rows:
        model.zero_grad(set_to_none=True)
        batch = _completion_batch(
            tokenizer,
            [record.prompt],
            [str(record.target)],
            config.device,
            config.max_seq_len,
        )
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logps = _chosen_logprobs(outputs.logits, batch["input_ids"])
        loss = -_masked_mean(logps, batch["completion_mask"][:, 1:].float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite profile loss for row={record.example_id}")
        loss.backward()
        for name, parameter in model.named_parameters():
            layer = _layer_from_name(name)
            if (
                parameter.grad is None
                or parameter.grad.ndim != 2
                or layer not in selected_set
                or not name.endswith(".weight")
                or not any(name.endswith(f"{suffix}.weight") for suffix in config.target_suffixes)
            ):
                continue
            gradient = parameter.grad.detach().float().cpu()
            layer_pressure[layer] = layer_pressure.get(layer, 0.0) + float(gradient.norm().item())
            rank = min(config.profile_rank, gradient.shape[0], gradient.shape[1])
            if rank <= 0 or float(gradient.norm()) == 0.0:
                continue
            omega = torch.randn(gradient.shape[1], rank, generator=generator)
            basis, _ = torch.linalg.qr(gradient @ omega, mode="reduced")
            basis_parts.setdefault(name, []).append(basis)
    if not basis_parts or not layer_pressure:
        raise RuntimeError("profile-only event produced no target-module gradient geometry")
    total_pressure = sum(layer_pressure.values())
    covered = 0.0
    selected_layers: List[int] = []
    minimum = min(8, len(layer_pressure))
    for layer, pressure in sorted(layer_pressure.items(), key=lambda item: (-item[1], item[0])):
        selected_layers.append(layer)
        covered += pressure
        if len(selected_layers) >= minimum and covered / max(total_pressure, 1e-12) >= 0.8:
            break
    selected_layer_set = set(selected_layers)
    tensors: Dict[str, Any] = {}
    names: Dict[str, str] = {}
    total_rank = 0
    for name, parts in sorted(basis_parts.items()):
        if _layer_from_name(name) not in selected_layer_set:
            continue
        merged, _ = torch.linalg.qr(torch.cat(parts, dim=1), mode="reduced")
        merged = merged[:, : min(config.profile_rank, merged.shape[1])].contiguous()
        key = f"basis_{len(tensors):04d}"
        tensors[key] = merged
        names[key] = name
        total_rank += merged.shape[1]
    from safetensors.torch import save_file

    provisional = blob_root / f".{safe_name(event.event_key)}.tmp.safetensors"
    save_file(tensors, str(provisional))
    digest = sha256_file(provisional)
    final = blob_root / f"{digest}.safetensors"
    if final.exists():
        provisional.unlink()
    else:
        provisional.replace(final)
    metadata_path = final.with_suffix(".json")
    if not metadata_path.exists():
        atomic_write_json(
            metadata_path,
            {
                "schema": "gradient_basis_v1",
                "parameters": names,
                "selected_layers": selected_layers,
                "pressure_coverage": covered / max(total_pressure, 1e-12),
            },
        )
    model.zero_grad(set_to_none=True)
    return final, digest, total_rank, tuple(sorted(selected_layers))


def _evaluate_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[ExampleRecord],
    event: LearningEvent,
    device: str,
    max_seq_len: int,
) -> Dict[str, float]:
    verifier = build_verifier(event.verifier) if event.verifier else None
    success = 0
    valid = 0
    stale = 0
    for index, record in enumerate(records):
        completion = _greedy_generate(model, tokenizer, record.prompt, device, max_new_tokens=min(64, max_seq_len))
        if verifier is not None:
            trajectory = Trajectory(
                event_key=event.event_key,
                example_id=record.example_id,
                group_id="eval",
                rollout_id=f"eval-{index}",
                policy_version=PolicyVersion(0),
                prompt=record.prompt,
                completion=completion,
            )
            result = score_sync(verifier, record, trajectory.verifier_view())
            success += int(result.success)
            valid += int(result.valid)
            stale += int(result.stale)
        elif record.target is not None:
            prediction = completion.strip().casefold()
            target = record.target.strip().casefold()
            success += int(prediction == target)
            valid += int(bool(prediction))
    count = max(1, len(records))
    return {
        "capability": success / count,
        "success_rate": success / count,
        "valid_rate": valid / count,
        "stale_rate": stale / count,
    }


def _greedy_generate(model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    torch = _torch()
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def _teacher_forced_loss(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[Tuple[str, str]],
    device: str,
    max_seq_len: int,
) -> float:
    torch = _torch()
    prompts = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    batch = _completion_batch(tokenizer, prompts, targets, device, max_seq_len)
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logps = _chosen_logprobs(outputs.logits, batch["input_ids"])
        loss = -_masked_mean(logps, batch["completion_mask"][:, 1:].float())
    return float(loss.item())


def _model_finite(model: Any) -> bool:
    torch = _torch()
    return all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())


def _save_consolidation_resume(
    destination: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    *,
    step: int,
    losses: Sequence[float],
    event_fingerprint: str,
    base_version: int,
) -> None:
    torch = _torch()
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "model").mkdir(parents=True)
    model.save_pretrained(temporary / "model", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "model")
    torch.save(
        {
            "step": int(step),
            "losses": list(losses),
            "event_fingerprint": event_fingerprint,
            "base_version": int(base_version),
            "optimizer": optimizer.state_dict(),
        },
        temporary / "resume.pt",
    )
    checksums = file_manifest(temporary, exclude_names=("checksums.json",))
    atomic_write_json(temporary / "checksums.json", checksums)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
