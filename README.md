# EvoXplain

EvoXplain is a domain-agnostic **evolutionary explainability engine**.

It takes a *sequence of trained models* (or training generations) from any
machine learning pipeline, extracts their feature contributions and structural
fingerprints, and tracks:

- how feature importance evolves over time,
- when the model undergoes structural drift, and
- how behaviour diverges across cohorts (e.g. ancestry, demographic groups).

The core design is deliberately **pipeline-agnostic**:

- EvoXplain can sit at the **end** of an omics pipeline (for example, after a
  tool ensemble learner), consuming model snapshots and
  returning evolutionary explainability metrics.
- The same engine can be applied to other domains such as housing prices,
  credit scoring, insurance risk, or any predictive system that is retrained
  over time.

## Core API (v0.1)

The main entry point is the `EvoXplainEngine` class:

```python
from evoxplain import EvoXplainEngine, ModelSnapshot

engine = EvoXplainEngine()

# Add model snapshots (e.g. successive training checkpoints or generations)
engine.add_snapshot(ModelSnapshot(
    generation=0,
    feature_importances={"feat1": 0.2, "feat2": 0.1},
    performance={"auc": 0.75}
))

engine.add_snapshot(ModelSnapshot(
    generation=1,
    feature_importances={"feat1": 0.25, "feat2": 0.08},
    performance={"auc": 0.78}
))

results = engine.run()

print(results.feature_trajectories)  # evolution of feature importances
print(results.feature_stability)     # simple stability scores
print(results.drift_scores)         # structural drift (if fingerprints were provided)
print(results.cohort_divergence)    # placeholder for future cohort analysis
```

This v0.1 implementation focuses on:

- building feature trajectories across generations,
- computing simple stability measures, and
- providing a hook for structural drift and cohort divergence analysis.

Future versions will extend these metrics and add plotting utilities and
domain-specific adapters.
