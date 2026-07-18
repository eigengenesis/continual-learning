# Contributing

Contributions should preserve the repository's central experimental invariant:
the committed model must remain unchanged until a candidate has passed its
declared capability, retention, general, numerical, access, and budget gates.

## Development Setup

```bash
git clone https://github.com/eigengenesis/continual-learning.git
cd continual-learning
python -m pip install -e ".[dev]"
python -m pytest -q
```

GPU-specific work should document the model, accelerator, dtype, package
versions, manifest hash, seed, and exact command used.

## Pull Requests

- Keep changes focused and explain the research or engineering question.
- Add tests for new event contracts, verifier behavior, geometry decisions,
  resume boundaries, or commit gates.
- Do not include checkpoints, raw datasets, notebook caches, generated plots,
  or authentication credentials.
- Preserve verifier privacy: reward targets must not enter trajectories,
  optimizer inputs, acquisition artifacts, or consolidation records.
- Preserve replay auditing: update code may access only the current event's
  frozen training rows.
- Treat changed hyperparameters or gates as a new event revision rather than
  silently mutating an existing experiment.

## Reporting Results

Separate infrastructure validation from model-level evidence. A locally passing
test suite establishes software invariants; it does not establish learning,
retention, or scaling behavior. Report failed and rejected runs alongside
successful runs when they affect the interpretation.
