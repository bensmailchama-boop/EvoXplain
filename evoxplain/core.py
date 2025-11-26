from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ModelSnapshot:
    """
    Single model instance at a given 'generation' or time step.

    Parameters
    ----------
    generation : int
        Index of the generation or training iteration this model belongs to.
    feature_importances : Dict[str, float]
        Mapping from feature name to importance score for this model.
    performance : Dict[str, float]
        Mapping from metric name to value (e.g. {"auc": 0.81, "loss": 0.39}).
    structure_fingerprint : Optional[np.ndarray]
        Optional vector representation of the model's internal structure
        (for example flattened weights, an embedding, or any fixed-length
        fingerprint). Used for structural drift analysis.
    cohort_importances : Optional[Dict[str, Dict[str, float]]]
        Optional mapping from cohort label (e.g. "EUR", "AFR") to a dictionary
        of feature importances for that cohort.
    """
    generation: int
    feature_importances: Dict[str, float]
    performance: Dict[str, float]
    structure_fingerprint: Optional[np.ndarray] = None
    cohort_importances: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class EvoXplainResult:
    """
    Container for computed evolutionary explainability outputs.

    Attributes
    ----------
    feature_trajectories : Dict[str, List[float]]
        For each feature, the sequence of importance values across generations.
    feature_stability : Dict[str, float]
        For each feature, a stability score derived from its trajectory.
    drift_scores : List[float]
        Structural drift scores between successive generations.
    cohort_divergence : Dict[str, Any]
        Container for any cohort-level divergence metrics.
    """
    feature_trajectories: Dict[str, List[float]]
    feature_stability: Dict[str, float]
    drift_scores: List[float]
    cohort_divergence: Dict[str, Any]


class EvoXplainEngine:
    """
    Core EvoXplain engine.

    This engine collects snapshots of models trained over time
    (or over evolutionary generations) and computes:

      * feature trajectories across generations
      * feature stability / volatility measures
      * structural drift between model versions
      * cohort-level divergence metrics (optional)

    The design is domain-agnostic: the engine can be plugged into
    any pipeline that can export feature importances and, optionally,
    some representation of model structure and cohort-specific
    attributions.
    """

    def __init__(self):
        self.snapshots: List[ModelSnapshot] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def add_snapshot(self, snapshot: ModelSnapshot) -> None:
        """
        Add a new trained model snapshot.

        Parameters
        ----------
        snapshot : ModelSnapshot
            The snapshot to add. Snapshots will be kept ordered by
            their generation index.
        """
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.generation)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------
    def compute_feature_trajectories(self) -> Dict[str, List[float]]:
        """
        Build feature trajectories over generations.

        Returns
        -------
        Dict[str, List[float]]
            A mapping from feature name to a list of importance values,
            ordered by the generation indices of the stored snapshots.
            Features that are absent in a given snapshot receive 0.0
            for that generation.
        """
        # Collect all feature names across all snapshots
        all_features = set()
        for s in self.snapshots:
            all_features.update(s.feature_importances.keys())

        trajectories: Dict[str, List[float]] = {f: [] for f in all_features}

        for s in self.snapshots:
            for f in all_features:
                val = s.feature_importances.get(f, 0.0)
                trajectories[f].append(val)

        return trajectories

    def compute_feature_stability(
        self,
        trajectories: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute a simple stability score for each feature trajectory.

        For v0.1, we use a basic 1 / (1 + variance) transform,
        where higher values indicate more stability.

        Parameters
        ----------
        trajectories : Dict[str, List[float]]
            Feature trajectories as returned by `compute_feature_trajectories`.

        Returns
        -------
        Dict[str, float]
            A mapping from feature name to stability score.
        """
        stability: Dict[str, float] = {}
        for f, series in trajectories.items():
            arr = np.asarray(series, dtype=float)
            var = float(np.var(arr))
            stability[f] = 1.0 / (1.0 + var)
        return stability

    def compute_structural_drift(self) -> List[float]:
        """
        Compute structural drift scores between successive snapshots.

        For v0.1, this is the L2 distance between consecutive
        structure_fingerprint vectors, ignoring snapshots that do not
        provide a fingerprint.

        Returns
        -------
        List[float]
            A list of drift scores between successive fingerprints.
        """
        drift_scores: List[float] = []

        # Filter snapshots that have a structure_fingerprint
        structs = [
            s.structure_fingerprint for s in self.snapshots
            if s.structure_fingerprint is not None
        ]
        if len(structs) < 2:
            return drift_scores

        for prev, curr in zip(structs[:-1], structs[1:]):
            diff = float(np.linalg.norm(curr - prev))
            drift_scores.append(diff)

        return drift_scores

    def compute_cohort_divergence(self) -> Dict[str, Any]:
        """
        Placeholder for cohort-level divergence analysis.

        This method is intended to compare feature trajectories or
        importance profiles across different cohorts (for example,
        ancestry groups, demographic groups, or other stratifications).

        For v0.1 this returns an empty dictionary to be fleshed out
        in future versions.
        """
        divergence: Dict[str, Any] = {}
        # TODO: implement cohort analysis in future versions
        return divergence

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def run(self) -> EvoXplainResult:
        """
        Run the full EvoXplain pipeline and return all metrics.

        Returns
        -------
        EvoXplainResult
            Aggregated feature trajectories, stability scores,
            structural drift scores, and cohort divergence metrics.
        """
        trajectories = self.compute_feature_trajectories()
        stability = self.compute_feature_stability(trajectories)
        drift = self.compute_structural_drift()
        cohorts = self.compute_cohort_divergence()

        return EvoXplainResult(
            feature_trajectories=trajectories,
            feature_stability=stability,
            drift_scores=drift,
            cohort_divergence=cohorts,
        )
