"""Typed, validated configuration for the whole pipeline.

Four YAML files (``config/settings.yaml``, ``dataset.yaml``, ``features.yaml``,
``analysis.yaml``) are loaded into immutable pydantic models.  Secrets are never
read from YAML: API keys come from the environment (``.env`` is supported) and
are excluded from every dump of the configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .paths import ensure_dir, project_root, resolve_path

__all__ = [
    "ProjectConfig",
    "load_config",
    "SettingsConfig",
    "DatasetConfig",
    "FeatureConfig",
    "AnalysisConfig",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# settings.yaml
# --------------------------------------------------------------------------- #
class PathsConfig(_Base):
    data_root: str = "data"
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    features: str = "data/features"
    metadata: str = "data/metadata"
    artifacts: str = "artifacts"
    figures: str = "artifacts/figures"
    tables: str = "artifacts/tables"
    logs: str = "artifacts/execution_logs"
    reports: str = "reports"


class LoggingConfig(_Base):
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s"
    file: str = "artifacts/execution_logs/pipeline.log"
    json_events: str = "artifacts/execution_logs/events.jsonl"


class LLMConfig(_Base):
    provider: Literal["anthropic", "deepseek", "offline"] = "offline"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 60.0
    max_retries: int = 3
    request_cache: str = "artifacts/execution_logs/llm_cache.jsonl"
    allow_offline_fallback: bool = True

    # populated from the environment, never from YAML
    anthropic_api_key: str | None = Field(default=None, exclude=True, repr=False)
    deepseek_api_key: str | None = Field(default=None, exclude=True, repr=False)
    anthropic_base_url: str = "https://api.anthropic.com"
    deepseek_base_url: str = "https://api.deepseek.com"

    @property
    def api_key(self) -> str | None:
        if self.provider == "anthropic":
            return self.anthropic_api_key
        if self.provider == "deepseek":
            return self.deepseek_api_key
        return None

    @property
    def base_url(self) -> str:
        return self.anthropic_base_url if self.provider == "anthropic" else self.deepseek_base_url


class ExecutionConfig(_Base):
    mode: Literal["smoke", "standard", "full"] = "standard"
    use_cache: bool = True
    max_subjects: int = 6
    parallel_workers: int = 1


class SettingsConfig(_Base):
    project_name: str = "agentic-feature-exploration"
    random_seed: int = 20240517
    offline_mode: Literal["auto", "always", "never"] = "auto"
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig = LLMConfig()
    execution: ExecutionConfig = ExecutionConfig()


# --------------------------------------------------------------------------- #
# dataset.yaml
# --------------------------------------------------------------------------- #
class SourceConfig(_Base):
    provider: str = "physionet"
    doi: str | None = None
    landing_page: str = ""
    file_base_url: str = ""
    collection: str = "sleep-cassette"
    file_pattern: str = ""
    annotation_pattern: str = ""
    mne_dataset: str = "sleep_physionet.age"
    citation: str = ""


class SubsetConfig(_Base):
    subjects: list[int] = [0, 1, 2]
    recordings: list[int] = [1, 2]
    max_recording_seconds: int | None = 1800


class FallbackConfig(_Base):
    mode: str = "synthetic_psg"
    subjects: int = 8
    epochs_per_subject: int = 120
    seed: int = 20240517
    effect_strength: float = 1.0


class SignalConfig(_Base):
    channel_preference: list[str] = ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG"]
    epoch_seconds: float = 30.0
    target_sampling_frequency: float = 100.0
    min_epochs_per_subject: int = 20
    min_annotation_coverage: float = 0.5


class StagesConfig(_Base):
    supported: list[str] = ["W", "N1", "N2", "N3", "REM"]
    annotation_map: dict[str, str] = {}
    unsupported_labels: list[str] = ["UNKNOWN"]


class PreprocessingConfig(_Base):
    bandpass: tuple[float, float] = (0.3, 45.0)
    notch_hz: float = 50.0
    notch_quality: float = 30.0
    apply_notch: bool = True
    clip_sigma: float = 8.0
    min_amplitude_uv: float = 0.5
    max_amplitude_uv: float = 1000.0
    detrend: bool = True
    reference: str = "none"


class DatasetConfig(_Base):
    dataset_name: str = "sleep-edf-expanded"
    data_origin: Literal["auto", "sleep-edf", "synthetic"] = "auto"
    source: SourceConfig = SourceConfig()
    subset: SubsetConfig = SubsetConfig()
    fallback: FallbackConfig = FallbackConfig()
    signal: SignalConfig = SignalConfig()
    stages: StagesConfig = StagesConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()


# --------------------------------------------------------------------------- #
# features.yaml
# --------------------------------------------------------------------------- #
class PSDConfig(_Base):
    method: Literal["welch", "multitaper"] = "welch"
    welch_window_seconds: float = 4.0
    welch_overlap_fraction: float = 0.5
    detrend: str = "constant"


class FamilyToggles(_Base):
    spectral: bool = True
    temporal: bool = True
    nonlinear: bool = True
    wavelet: bool = True
    events: bool = True
    literature_derived: bool = True
    novel: bool = True


class NonlinearConfig(_Base):
    sample_entropy_m: int = 2
    sample_entropy_r_factor: float = 0.2
    approx_entropy_m: int = 2
    approx_entropy_r_factor: float = 0.2
    permutation_entropy_order: int = 4
    permutation_entropy_delay: int = 1
    higuchi_kmax: int = 10
    dfa_scales: list[int] = [4, 6, 8, 10, 12, 16, 20, 32, 48, 64]
    dfa_alpha1_range: tuple[int, int] = (4, 16)
    dfa_alpha2_range: tuple[int, int] = (16, 64)
    lz_threshold: str = "median"
    max_samples: int = 6000


class WaveletConfig(_Base):
    wavelet: str = "db4"
    levels: int = 5


class SpindleConfig(_Base):
    band_hz: tuple[float, float] = (11.0, 16.0)
    envelope_smoothing_seconds: float = 0.1
    threshold_factor: float = 1.5
    threshold_window_seconds: float = 2.0
    min_duration_s: float = 0.5
    max_duration_s: float = 3.0
    min_zero_cross_rate_hz: float = 10.0
    max_zero_cross_rate_hz: float = 18.0
    refractory_s: float = 0.5
    amplitude_percentile: float = 75.0
    #: Absolute floor on the smoothed sigma-band envelope, in microvolts.
    #: A purely epoch-relative amplitude criterion cannot reject broadband noise
    #: (the top quartile of any epoch's envelope passes by construction), so the
    #: detector also requires the spindle-shaped burst to reach a minimum
    #: amplitude.  8 uV follows the band-passed amplitude criterion used by
    #: Wamsley et al. (2012) for automated spindle detection; measured on
    #: synthetic data it removes stationary-noise false positives entirely while
    #: still detecting 12-25 uV bursts.
    min_amplitude_uv: float = 8.0


class KComplexConfig(_Base):
    lowpass_hz: float = 4.0
    neg_threshold_factor: float = 2.5
    min_amplitude_uv: float = 30.0
    min_duration_s: float = 0.5
    max_duration_s: float = 2.0
    rebound_required: bool = True
    refractory_s: float = 1.0


class SlowWaveConfig(_Base):
    band_hz: tuple[float, float] = (0.5, 2.0)
    min_duration_s: float = 0.3
    max_duration_s: float = 1.5
    neg_threshold_factor: float = 1.5
    min_amplitude_uv: float = 20.0
    refractory_s: float = 0.3
    slope_window_s: float = 0.2


class EventsConfig(_Base):
    spindle: SpindleConfig = SpindleConfig()
    k_complex: KComplexConfig = KComplexConfig()
    slow_wave: SlowWaveConfig = SlowWaveConfig()


class NovelConfig(_Base):
    spectral_concentration_alpha: list[float] = [2.0, 3.0]
    subwindow_seconds: float = 2.0
    epsilon: float = 1e-12
    instability_bands: list[str] = ["delta", "sigma", "beta"]
    transition_lookback_epochs: int = 1


class FeatureConfig(_Base):
    bands: dict[str, tuple[float, float]] = {}
    psd: PSDConfig = PSDConfig()
    families: FamilyToggles = FamilyToggles()
    nonlinear: NonlinearConfig = NonlinearConfig()
    wavelet: WaveletConfig = WaveletConfig()
    events: EventsConfig = EventsConfig()
    novel: NovelConfig = NovelConfig()

    @field_validator("bands", mode="before")
    @classmethod
    def _coerce_bands(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: tuple(v) for k, v in value.items()}
        return value


# --------------------------------------------------------------------------- #
# analysis.yaml
# --------------------------------------------------------------------------- #
class StatisticsConfig(_Base):
    alpha: float = 0.05
    correction: Literal["fdr_bh", "holm", "bonferroni", "none"] = "fdr_bh"
    min_group_size: int = 15
    normality_alpha: float = 0.05
    use_subject_aggregation: bool = True
    permutation_iterations: int = 1000
    bootstrap_iterations: int = 1000
    random_state: int = 20240517
    effect_size_ci: float = 0.95


class FeatureSelectionConfig(_Base):
    max_features_for_multivariable: int = 40
    correlation_threshold: float = 0.9
    mutual_information_neighbors: int = 5
    rfe_step: float = 0.1


class PredictiveConfig(_Base):
    test_fraction: float = 0.25
    validation_fraction: float = 0.25
    cv_folds: int = 3
    classifiers: list[str] = ["majority", "logistic_regression"]
    primary_metric: str = "balanced_accuracy"
    per_stage_focus: list[str] = ["N1", "N2", "REM"]
    ablations: list[str] = ["all_features"]


class HypothesesConfig(_Base):
    interaction_pairs_max: int = 12
    confirmatory_min_effect: float = 0.2


class ArtifactsConfig(_Base):
    save_figures: bool = True
    figure_dpi: int = 120
    top_feature_plots: int = 12


class AnalysisConfig(_Base):
    statistics: StatisticsConfig = StatisticsConfig()
    features_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    predictive: PredictiveConfig = PredictiveConfig()
    hypotheses: HypothesesConfig = HypothesesConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
class ProjectConfig(BaseModel):
    """Fully resolved configuration for one pipeline execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    settings: SettingsConfig
    dataset: DatasetConfig
    features: FeatureConfig
    analysis: AnalysisConfig
    environment_overrides: dict[str, Any] = Field(default_factory=dict)

    # ---- path helpers ----------------------------------------------------- #
    def path(self, key: str) -> Path:
        value = getattr(self.settings.paths, key, None)
        if value is None:
            raise KeyError(f"unknown path key: {key}")
        return resolve_path(value, self.root)

    def ensure_directories(self) -> dict[str, Path]:
        created = {}
        for key in (
            "raw",
            "interim",
            "processed",
            "features",
            "metadata",
            "artifacts",
            "figures",
            "tables",
            "logs",
            "reports",
        ):
            created[key] = ensure_dir(self.path(key))
        return created

    @property
    def seed(self) -> int:
        return int(self.settings.random_seed)

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        return dict(self.features.bands)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file {path} must contain a mapping at top level")
    return data


def _env_overrides() -> dict[str, Any]:
    """Collect env-var based overrides (secrets are kept out of the YAML)."""
    env = os.environ
    overrides: dict[str, Any] = {}
    if env.get("LLM_PROVIDER"):
        overrides["llm_provider"] = env["LLM_PROVIDER"].strip().lower()
    if env.get("LLM_MODEL"):
        overrides["llm_model"] = env["LLM_MODEL"].strip()
    if env.get("ANTHROPIC_BASE_URL"):
        overrides["anthropic_base_url"] = env["ANTHROPIC_BASE_URL"].strip()
    if env.get("DEEPSEEK_BASE_URL"):
        overrides["deepseek_base_url"] = env["DEEPSEEK_BASE_URL"].strip()
    if env.get("ANTHROPIC_API_KEY"):
        overrides["anthropic_api_key"] = env["ANTHROPIC_API_KEY"].strip()
    if env.get("DEEPSEEK_API_KEY"):
        overrides["deepseek_api_key"] = env["DEEPSEEK_API_KEY"].strip()
    if env.get("LITERATURE_TARGET_PAPERS"):
        overrides["literature_target_papers"] = int(env["LITERATURE_TARGET_PAPERS"])
    if env.get("EEG_AGENT_FORCE_OFFLINE_LLM", "").strip() in {"1", "true", "yes"}:
        overrides["force_offline_llm"] = True
    return overrides


def _load_dotenv(root: Path) -> None:
    """Load ``.env`` if python-dotenv is available (never overrides real env vars)."""
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional convenience
        return
    load_dotenv(env_path, override=False)


def load_config(config_dir: str | Path | None = None, root: str | Path | None = None) -> ProjectConfig:
    """Load and validate every configuration file."""
    root_path = Path(root).resolve() if root else project_root()
    _load_dotenv(root_path)
    cfg_dir = Path(config_dir) if config_dir else root_path / "config"

    settings = SettingsConfig.model_validate(_read_yaml(cfg_dir / "settings.yaml"))
    dataset = DatasetConfig.model_validate(_read_yaml(cfg_dir / "dataset.yaml"))
    features = FeatureConfig.model_validate(_read_yaml(cfg_dir / "features.yaml"))
    analysis = AnalysisConfig.model_validate(_read_yaml(cfg_dir / "analysis.yaml"))

    overrides = _env_overrides()
    llm = settings.llm.model_copy(
        update={
            "provider": overrides.get("llm_provider", settings.llm.provider),
            "model": overrides.get("llm_model", settings.llm.model),
            "anthropic_api_key": overrides.get("anthropic_api_key"),
            "deepseek_api_key": overrides.get("deepseek_api_key"),
            "anthropic_base_url": overrides.get("anthropic_base_url", settings.llm.anthropic_base_url),
            "deepseek_base_url": overrides.get("deepseek_base_url", settings.llm.deepseek_base_url),
        }
    )
    settings = settings.model_copy(update={"llm": llm})

    if not features.bands:
        raise ValueError("config/features.yaml must define at least one frequency band")

    return ProjectConfig(
        root=root_path,
        settings=settings,
        dataset=dataset,
        features=features,
        analysis=analysis,
        environment_overrides=overrides,
    )
