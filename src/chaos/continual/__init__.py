"""Event-driven, transactional continual-learning infrastructure."""

from .acquisition import (
    AcquisitionConfig,
    DemonstrationAcquirer,
    OnPolicyAcquirer,
    PolicyUpdateConfig,
    acquire_from_reward,
)
from .algorithms import (
    DemonstrationAlgorithm,
    GroupRelativeAlgorithm,
    HybridAlgorithm,
    ReferenceDistillationAlgorithm,
)
from .artifacts import AcquisitionArtifact, CandidateArtifact
from .commit_store import CurrentVersion, TransactionStore
from .engine import ContinualLearningEngine
from .evaluator import CommitEvaluator, EvaluationReport, SystemChecks
from .events import (
    AcquisitionBudget,
    ContextSpec,
    DatasetRef,
    ExampleRecord,
    GateBundle,
    GateRule,
    LearningEvent,
    PublicExample,
    TargetRef,
    VerifierSpec,
)
from .router import LearningSignalRouter, RoutingDecision
from .geometry import (
    CapacityReport,
    GeometryController,
    GeometryDecision,
    GeometryMeasurement,
    GeometryPlan,
    LayerMeasurement,
)
from .profiles import ProfileRecord, ProfileRegistry, ProfileTensorStore
from .hf_runtime import HuggingFaceContinualRuntime, HuggingFaceRuntimeConfig
from .runtime import TabularContinualRuntime, TabularTemporaryPolicy
from .stream import DirectoryEventSource, EventLease
from .trajectories import PolicyVersion, RolloutGroup, SamplingConfig, Trajectory, TrainingSample
from .verifiers import RewardResult, Verifier, build_verifier, register_verifier

__all__ = [
    "AcquisitionBudget",
    "AcquisitionConfig",
    "AcquisitionArtifact",
    "CandidateArtifact",
    "CapacityReport",
    "ContinualLearningEngine",
    "CommitEvaluator",
    "ContextSpec",
    "CurrentVersion",
    "DatasetRef",
    "DemonstrationAlgorithm",
    "DemonstrationAcquirer",
    "ExampleRecord",
    "GateBundle",
    "GateRule",
    "GeometryController",
    "GeometryDecision",
    "GeometryMeasurement",
    "GeometryPlan",
    "GroupRelativeAlgorithm",
    "HuggingFaceContinualRuntime",
    "HuggingFaceRuntimeConfig",
    "HybridAlgorithm",
    "LearningEvent",
    "LearningSignalRouter",
    "LayerMeasurement",
    "PolicyVersion",
    "PolicyUpdateConfig",
    "ProfileRecord",
    "ProfileRegistry",
    "ProfileTensorStore",
    "PublicExample",
    "ReferenceDistillationAlgorithm",
    "RewardResult",
    "RolloutGroup",
    "RoutingDecision",
    "SamplingConfig",
    "TargetRef",
    "TabularContinualRuntime",
    "TabularTemporaryPolicy",
    "Trajectory",
    "TrainingSample",
    "Verifier",
    "VerifierSpec",
    "DirectoryEventSource",
    "EventLease",
    "TransactionStore",
    "EvaluationReport",
    "OnPolicyAcquirer",
    "SystemChecks",
    "acquire_from_reward",
    "build_verifier",
    "register_verifier",
]
