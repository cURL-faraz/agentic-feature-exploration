"""Statistical analysis engine.

Modules
-------
``descriptive``        distribution summaries, epoch- and subject-level
``hypothesis_tests``   assumption checks, decision-table test selection, post-hoc
``effect_sizes``       Cohen's d / Hedges' g / rank-based measures + bootstrap CIs
``multiple_testing``   Benjamini-Hochberg, Holm, Bonferroni
``correlations``       order-free association measures for the nominal stage label
``feature_selection``  redundancy control, rankings, bounded interaction search
``engine``             orchestration + persistence of the whole stage
"""

from .correlations import feature_stage_associations, multiclass_association
from .descriptive import describe_feature_by_stage, describe_table, subject_stage_means
from .effect_sizes import cohens_d, hedges_g, hedges_g_with_ci, rank_biserial
from .engine import StatisticalAnalysis, run_statistical_analysis, stage_pairs
from .feature_selection import prune_redundant, prune_redundant as redundancy_prune
from .hypothesis_tests import compare_feature, compare_feature as compare_feature_across_stages
from .multiple_testing import correct_pvalues

__all__ = [
    "StatisticalAnalysis",
    "run_statistical_analysis",
    "stage_pairs",
    "compare_feature",
    "compare_feature_across_stages",
    "describe_table",
    "describe_feature_by_stage",
    "subject_stage_means",
    "correct_pvalues",
    "cohens_d",
    "hedges_g",
    "hedges_g_with_ci",
    "rank_biserial",
    "feature_stage_associations",
    "multiclass_association",
    "prune_redundant",
    "redundancy_prune",
]
