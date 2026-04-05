"""Typed configuration access for the housing model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.core.paths import CONFIG_DIR


load_dotenv()


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    base_year: int
    end_year: int
    date_frequency: str
    regions: list[str]


@dataclass(frozen=True)
class ArtifactConfig:
    canonical_panel_file: str
    stage4_parameters_file: str
    stage5_summary_file: str
    stage6_handoff_file: str
    stage7_snapshot_file: str
    stage7_metadata_file: str


@dataclass(frozen=True)
class ModelConfig:
    sde_type: str
    dt: float
    n_simulations: int
    drift_mu: float | None = None
    mean_reversion_theta: float | None = None
    diffusion_sigma: float | None = None
    long_run_mean: float | None = None
    rho_momentum: float | None = None
    r_squared_within: float | None = None
    calibration_date: str | None = None
    calibration_version: str | None = None
    endogeneity_corrected: bool | None = None


@dataclass(frozen=True)
class ThresholdConfig:
    supportive: float
    mixed: float
    cautious: float = 0.0


@dataclass(frozen=True)
class ComponentWeights:
    base_model: float
    affordability: float
    downside_risk: float


@dataclass(frozen=True)
class ReitWeights:
    base_model: float
    yield_signal: float
    downside_risk: float


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    display_name: str
    rate_shift_bps: float
    income_growth_shift_pct: float
    supply_shift_pct: float
    volatility_multiplier: float
    drift_shift_pct: float
    valuation_gap_closure: float
    financial_stress_shift: float
    transaction_shift: float
    min_target_price_ratio: float
    regime_prob_normal: float
    regime_prob_stress: float
    regime_prob_boom: float
    note: str


@dataclass(frozen=True)
class UIConfig:
    deployment: str
    default_region: str
    default_income_gbp: int
    default_deposit_gbp: int
    default_mortgage_term_years: int
    default_target_yield_pct: float
    default_holding_horizon_years: int
    default_property_price_gbp: int
    historical_mode_label: str
    scenario_labels: list[str]
    scenario_percentiles: list[int]
    fan_chart_percentiles: list[int]
    risk_tolerance_options: list[str]


@dataclass(frozen=True)
class ControlsConfig:
    override_warning_rate_bps: float
    override_warning_sigma_multiplier: float
    override_warning_horizon_years: float
    max_paths_interactive: int
    default_paths_interactive: int
    random_seed: int
    minimum_price_floor_ratio: float
    affordability_multiple: float


@dataclass(frozen=True)
class FreshnessConfig:
    historical_only: bool
    max_live_data_age_months: int
    stale_warning_months: int
    disclosure_text: str


@dataclass(frozen=True)
class FairValueConfig:
    anchor_months: list[int]
    regressors: list[str]
    kappa_fallback_phi: float
    kappa_phi_upper_bound: float
    sigma_floor_annual: float
    sigma_cap_annual: float
    financial_stress_activation_threshold: float
    # Per-region sigma cap overrides (region name → annual cap). Keys starting with "_" are
    # documentation notes and are filtered out in from_dict(). Falls back to sigma_cap_annual.
    sigma_cap_overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    baseline_target_weight: float
    baseline_volatility_blend: float
    stress_sigma_multiplier_cap: float


@dataclass(frozen=True)
class ValidationConfig:
    canonical_runner_name: str
    forecast_horizons_months: list[int]
    score_backtest_horizons_months: list[int]


@dataclass(frozen=True)
class GuardrailsConfig:
    beta_rate_max: float
    beta_financial_stress_max: float
    beta_income_min: float
    beta_transaction_min: float
    rho_max: float
    gamma_min_annual: float
    gamma_max_annual: float
    sigma_raw_weight: float
    sigma_winsor_weight: float
    kappa_fallback_mode: str


@dataclass(frozen=True)
class ScoringConfig:
    thresholds: ThresholdConfig
    consumer_weights: dict[str, float]
    reit_weights: dict[str, float]
    app_consumer_weights: ComponentWeights
    app_reit_weights: ReitWeights
    scenario_consumer_weights: dict[str, float]
    scenario_reit_weights: dict[str, float]


@dataclass(frozen=True)
class HousingModelConfig:
    project: ProjectConfig
    artifacts: ArtifactConfig
    model: ModelConfig
    scoring: ScoringConfig
    ui: UIConfig
    controls: ControlsConfig
    freshness: FreshnessConfig
    fair_value: FairValueConfig
    simulation: SimulationConfig
    validation: ValidationConfig
    guardrails: GuardrailsConfig
    scenarios: dict[str, ScenarioPreset]

    def model_dump(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "HousingModelConfig":
        def _pick(data: dict, model_cls):
            allowed = set(model_cls.__annotations__.keys())
            return {key: value for key, value in data.items() if key in allowed}

        return cls(
            project=ProjectConfig(**_pick(raw["project"], ProjectConfig)),
            artifacts=ArtifactConfig(
                **_pick(raw["artifacts"], ArtifactConfig)),
            model=ModelConfig(**_pick(raw["model"], ModelConfig)),
            scoring=ScoringConfig(
                thresholds=ThresholdConfig(
                    **_pick(raw["scoring"]["thresholds"], ThresholdConfig)),
                consumer_weights=raw["scoring"]["consumer_weights"],
                reit_weights=raw["scoring"]["reit_weights"],
                app_consumer_weights=ComponentWeights(
                    **_pick(raw["scoring"]["app_consumer_weights"], ComponentWeights)),
                app_reit_weights=ReitWeights(
                    **_pick(raw["scoring"]["app_reit_weights"], ReitWeights)),
                scenario_consumer_weights={
                    k: float(v)
                    for k, v in raw["scoring"]["scenario_consumer_weights"].items()
                    if not k.startswith("_")
                },
                scenario_reit_weights={
                    k: float(v)
                    for k, v in raw["scoring"]["scenario_reit_weights"].items()
                    if not k.startswith("_")
                },
            ),
            ui=UIConfig(**_pick(raw["ui"], UIConfig)),
            controls=ControlsConfig(**_pick(raw["controls"], ControlsConfig)),
            freshness=FreshnessConfig(
                **_pick(raw["freshness"], FreshnessConfig)),
            fair_value=FairValueConfig(
                **_pick({k: v for k, v in raw["fair_value"].items() if k != "sigma_cap_overrides"}, FairValueConfig),
                sigma_cap_overrides={
                    k: float(v)
                    for k, v in raw["fair_value"].get("sigma_cap_overrides", {}).items()
                    if not k.startswith("_")
                },
            ),
            simulation=SimulationConfig(
                **_pick(raw["simulation"], SimulationConfig)),
            validation=ValidationConfig(
                **_pick(raw["validation"], ValidationConfig)),
            guardrails=GuardrailsConfig(
                **_pick(raw["guardrails"], GuardrailsConfig)),
            scenarios={key: ScenarioPreset(
                **_pick(value, ScenarioPreset)) for key, value in raw["scenarios"].items()},
        )


DEFAULT_CONFIG_PATH = CONFIG_DIR / "parameters.json"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> HousingModelConfig:
    """Load and validate the canonical project configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return HousingModelConfig.from_dict(raw)


def reset_config_cache() -> None:
    """Reset the cached config, mainly for tests."""
    load_config.cache_clear()
